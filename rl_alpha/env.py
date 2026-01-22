"""
Gym-Compatible RL Environment for Alpha Expression Generation

Implements an environment where:
- State: Current partial AST + backtest history embedding
- Action: Select next operator/operand from valid actions
- Reward: Multi-objective score from Brain API simulation

Reference: Alpha² + Chain-of-Alpha methodology
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import logging
import random

from .grammar import (
    ExpressionGrammar, 
    ASTNode, 
    Operator, 
    Operand,
    DimType,
    OPERATORS,
    build_expression_from_ast,
    create_default_operands
)

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Configuration for AlphaEnv."""
    max_depth: int = 6
    max_nodes: int = 12
    max_steps: int = 20
    reward_scale: float = 1.0
    invalid_action_penalty: float = -0.1
    incomplete_penalty: float = -0.5
    diversity_bonus_weight: float = 0.2
    use_brain_api: bool = True
    simulation_timeout: int = 60
    # Simulation settings
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    neutralization: str = "INDUSTRY"
    decay: int = 4
    truncation: float = 0.02
    test_period: str = "P2Y"


class AlphaEnv(gym.Env):
    """
    Gymnasium environment for RL-based alpha expression generation.
    
    The agent builds an expression tree step by step, selecting operators
    and operands to construct a valid Brain alpha expression.
    
    State Space:
        - Current AST encoding (flattened tree structure)
        - Historical backtest statistics
        - Pool correlation summary
        
    Action Space:
        - Discrete: index into list of valid actions
        - Actions include operators (with params) and operands
        
    Reward:
        - Terminal: Multi-objective score from simulation
        - Intermediate: Small shaping rewards for valid structure
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        config: EnvConfig = None,
        operands: List[Operand] = None,
        brain_session = None,
        alpha_pool: List[Dict] = None,
        score_function = None,
        render_mode: str = None
    ):
        """
        Initialize the alpha generation environment.
        
        Args:
            config: Environment configuration
            operands: Available data field operands
            brain_session: Brain API session for simulation
            alpha_pool: Existing alpha pool for diversity checking
            score_function: Custom scoring function
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.config = config or EnvConfig()
        self.brain_session = brain_session
        self.alpha_pool = alpha_pool or []
        self.score_function = score_function
        self.render_mode = render_mode
        
        # Initialize grammar
        self.operands = operands or []
        self.grammar = ExpressionGrammar(
            operators=OPERATORS,
            operands=self.operands,
            max_depth=self.config.max_depth,
            max_nodes=self.config.max_nodes
        )
        
        # Build action space
        self._build_action_space()
        
        # Observation space: encoded AST + history
        # AST encoding: (max_nodes, node_features)
        # Node features: [type_one_hot(4), op_id, dim_type, depth, is_filled]
        self.node_feature_dim = 4 + len(OPERATORS) + len(DimType) + 3
        self.observation_space = spaces.Dict({
            'ast': spaces.Box(
                low=-1, high=1,
                shape=(self.config.max_nodes, self.node_feature_dim),
                dtype=np.float32
            ),
            'cursor': spaces.Discrete(self.config.max_nodes),
            'depth': spaces.Discrete(self.config.max_depth + 1),
            'valid_action_mask': spaces.MultiBinary(len(self.all_actions))
        })
        
        # Episode state
        self.reset()
    
    def _build_action_space(self):
        """Build the discrete action space from grammar."""
        self.all_actions = []
        self.action_to_idx = {}
        
        # Add operand actions
        for operand in self.operands:
            action = ('operand', operand)
            self.action_to_idx[action] = len(self.all_actions)
            self.all_actions.append(action)
        
        # Add operator actions (with parameter variations)
        for op_name, op in OPERATORS.items():
            if op.params:
                param_combos = self.grammar._get_param_combinations(op.params, max_combos=5)
                for combo in param_combos:
                    # Create hashable key
                    combo_tuple = tuple(sorted(combo.items()))
                    action = ('operator', (op_name, combo_tuple))
                    self.action_to_idx[action] = len(self.all_actions)
                    self.all_actions.append(action)
            else:
                action = ('operator', (op_name, ()))
                self.action_to_idx[action] = len(self.all_actions)
                self.all_actions.append(action)
        
        # Add special actions
        self.all_actions.append(('done', None))  # Complete expression
        self.action_to_idx[('done', None)] = len(self.all_actions) - 1
        
        self.action_space = spaces.Discrete(len(self.all_actions))
        
        logger.info(f"Built action space with {len(self.all_actions)} actions "
                   f"({len(self.operands)} operands, "
                   f"{len(self.all_actions) - len(self.operands) - 1} operator variants)")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        # Episode state
        self.current_ast: Optional[ASTNode] = None
        self.node_stack: List[Tuple[ASTNode, int, int]] = []  # (node, arg_idx, depth)
        self.step_count = 0
        self.done = False
        self.expression = ""
        
        # Track partial tree construction
        self.pending_nodes: List[Tuple[Optional[ASTNode], int, Set[DimType]]] = []
        # Start with root needing to be filled, accepts any dimension
        self.pending_nodes.append((None, 0, {DimType.ANY}))
        
        # Metrics
        self.episode_reward = 0.0
        self.valid_actions_taken = 0
        self.invalid_actions_taken = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Take action to expand the expression tree.
        
        Args:
            action: Index into action space
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        
        # Get action details
        action_tuple = self.all_actions[action]
        action_type, action_value = action_tuple
        
        # Check if action is valid
        valid_mask = self._get_valid_action_mask()
        if not valid_mask[action]:
            # Invalid action penalty
            reward = self.config.invalid_action_penalty
            self.invalid_actions_taken += 1
            logger.debug(f"Invalid action: {action_type}")
            return self._get_observation(), reward, False, False, self._get_info()
        
        self.valid_actions_taken += 1
        
        # Handle action
        if action_type == 'done':
            # Complete the expression
            if self._is_complete():
                terminated = True
                reward = self._evaluate_expression()
            else:
                # Incomplete expression
                reward = self.config.incomplete_penalty
                terminated = True
        
        elif action_type == 'operand':
            # Add leaf node
            reward = self._add_operand(action_value)
        
        elif action_type == 'operator':
            # Add operator node
            op_name, params_tuple = action_value
            params = dict(params_tuple)
            op = OPERATORS[op_name]
            reward = self._add_operator(op, params)
        
        # Check termination conditions
        if self.step_count >= self.config.max_steps:
            truncated = True
            if not terminated:
                # Evaluate partial expression if any
                if self._is_complete():
                    reward = self._evaluate_expression()
                else:
                    reward = self.config.incomplete_penalty
        
        # Check if tree is complete
        if not self.pending_nodes and not terminated and not truncated:
            terminated = True
            reward = self._evaluate_expression()
        
        self.done = terminated or truncated
        self.episode_reward += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _add_operand(self, operand: Operand) -> float:
        """Add an operand (leaf node) to the tree."""
        if not self.pending_nodes:
            return self.config.invalid_action_penalty
        
        parent, arg_idx, required_dim = self.pending_nodes.pop()
        
        # Create leaf node
        node = ASTNode(
            node_type='operand',
            value=operand,
            dim_type=operand.dim_type
        )
        
        # Attach to parent if exists
        if parent is not None and parent.node_type == 'operator':
            parent.children[arg_idx] = node
        else:
            self.current_ast = node
        
        # Small positive reward for valid expansion
        return 0.01
    
    def _add_operator(self, op: Operator, params: Dict) -> float:
        """Add an operator node to the tree."""
        if not self.pending_nodes:
            return self.config.invalid_action_penalty
        
        parent, arg_idx, required_dim = self.pending_nodes.pop()
        
        # Create operator node with placeholder children
        children = [None] * op.arity
        
        # Add parameter children
        param_children = []
        for param_name, param_val in params.items():
            param_children.append(ASTNode(
                node_type='param',
                value=f"{param_name}={param_val}"
            ))
        
        node = ASTNode(
            node_type='operator',
            value=op,
            children=children + param_children
        )
        
        # Attach to parent
        if parent is not None and parent.node_type == 'operator':
            parent.children[arg_idx] = node
        else:
            self.current_ast = node
        
        # Add children as pending (in reverse order so they're popped in order)
        for i in range(op.arity - 1, -1, -1):
            required = op.input_types[i] if i < len(op.input_types) else {DimType.ANY}
            self.pending_nodes.append((node, i, required))
        
        # Reward based on operator complexity
        return 0.02
    
    def _is_complete(self) -> bool:
        """Check if expression tree is complete."""
        return (self.current_ast is not None and 
                len(self.pending_nodes) == 0)
    
    def _evaluate_expression(self) -> float:
        """
        Evaluate the completed expression using Brain API.
        
        Returns:
            Multi-objective reward score
        """
        if self.current_ast is None:
            return self.config.incomplete_penalty
        
        # Build expression string
        try:
            self.expression = build_expression_from_ast(self.current_ast)
        except Exception as e:
            logger.warning(f"Failed to build expression: {e}")
            return self.config.incomplete_penalty
        
        if not self.expression:
            return self.config.incomplete_penalty
        
        logger.debug(f"Evaluating expression: {self.expression}")
        
        # If no Brain session, return structure-based score
        if not self.config.use_brain_api or self.brain_session is None:
            return self._compute_structure_score()
        
        # Simulate using Brain API
        try:
            return self._simulate_and_score()
        except Exception as e:
            logger.warning(f"Simulation failed: {e}")
            return self.config.incomplete_penalty
    
    def _compute_structure_score(self) -> float:
        """
        Compute a structure-based proxy score when Brain API is unavailable.
        
        Rewards:
        - Appropriate depth (not too shallow, not too deep)
        - Diversity from pool
        - Use of time-series operators
        """
        if self.current_ast is None:
            return 0.0
        
        score = 0.0
        
        # Depth bonus (optimal around 3-4)
        depth = self.current_ast.depth
        depth_score = 1.0 - abs(depth - 3.5) / 3.5
        score += 0.3 * max(0, depth_score)
        
        # Size bonus (optimal around 5-8 nodes)
        size = self.current_ast.size
        size_score = 1.0 - abs(size - 6) / 6
        score += 0.2 * max(0, size_score)
        
        # Diversity bonus
        if self.alpha_pool:
            max_sim = 0.0
            for existing in self.alpha_pool[-50:]:  # Check last 50
                if 'ast' in existing:
                    sim = self.grammar.compute_ast_similarity(
                        self.current_ast, existing['ast']
                    )
                    max_sim = max(max_sim, sim)
            diversity_bonus = 1.0 - max_sim
            score += self.config.diversity_bonus_weight * diversity_bonus
        
        return score * self.config.reward_scale
    
    def _simulate_and_score(self) -> float:
        """Run Brain API simulation and compute score."""
        import ace_lib as ace
        
        # Generate alpha config
        config = ace.generate_alpha(
            regular=self.expression,
            alpha_type="REGULAR",
            region=self.config.region,
            universe=self.config.universe,
            delay=self.config.delay,
            neutralization=self.config.neutralization,
            decay=self.config.decay,
            truncation=self.config.truncation,
            pasteurization="ON",
            test_period=self.config.test_period,
            unit_handling="VERIFY",
            nan_handling="ON",
            visualization=False
        )
        
        # Simulate
        results = ace.simulate_alpha_list_multi(
            self.brain_session,
            [config],
            limit_of_concurrent_simulations=1,
            limit_of_multi_simulations=1
        )
        
        if not results or not results[0].get('alpha_id'):
            return self.config.incomplete_penalty
        
        sim_result = results[0]
        
        # Compute score using scoring function
        if self.score_function:
            # Compute pool correlation
            pool_corr = self._compute_pool_correlation(sim_result)
            return self.score_function(sim_result, pool_corr=pool_corr)
        else:
            # Default scoring
            from alpha_scoring import calculate_alpha_score
            pool_corr = self._compute_pool_correlation(sim_result)
            return calculate_alpha_score(sim_result, prod_corr=0.0, self_corr=pool_corr)
    
    def _compute_pool_correlation(self, sim_result: Dict) -> float:
        """Compute maximum correlation with existing pool."""
        if not self.alpha_pool:
            return 0.0
        
        # Would need to call Brain API for actual correlation
        # For now, use AST similarity as proxy
        if self.current_ast is None:
            return 0.0
        
        max_sim = 0.0
        for existing in self.alpha_pool[-100:]:
            if 'ast' in existing:
                sim = self.grammar.compute_ast_similarity(
                    self.current_ast, existing['ast']
                )
                max_sim = max(max_sim, sim)
        
        return max_sim
    
    def _get_valid_action_mask(self) -> np.ndarray:
        """Get mask of currently valid actions."""
        mask = np.zeros(len(self.all_actions), dtype=np.int8)
        
        # If no pending nodes, only 'done' is valid
        if not self.pending_nodes:
            done_idx = self.action_to_idx.get(('done', None))
            if done_idx is not None:
                mask[done_idx] = 1
            return mask
        
        # Get current context
        parent, arg_idx, required_dim = self.pending_nodes[-1]
        current_depth = self._get_current_depth()
        current_size = self.current_ast.size if self.current_ast else 0
        
        # Check each action
        for i, action in enumerate(self.all_actions):
            action_type, action_value = action
            
            if action_type == 'done':
                # Done is valid only if tree is complete
                mask[i] = 1 if self._is_complete() else 0
            
            elif action_type == 'operand':
                # Check dimension compatibility
                operand = action_value
                if DimType.ANY in required_dim or operand.dim_type in required_dim:
                    mask[i] = 1
            
            elif action_type == 'operator':
                # Check depth constraint and dimension
                op_name, _ = action_value
                op = OPERATORS[op_name]
                
                if current_depth < self.config.max_depth - 1:
                    # Check output dimension compatibility
                    output_dim = op.get_output_dim([])
                    if DimType.ANY in required_dim or output_dim in required_dim:
                        mask[i] = 1
        
        return mask
    
    def _get_current_depth(self) -> int:
        """Get current depth in the tree."""
        if self.current_ast is None:
            return 0
        return len(self.pending_nodes)
    
    def _get_observation(self) -> Dict:
        """Get current observation."""
        # Encode AST
        ast_encoding = np.zeros(
            (self.config.max_nodes, self.node_feature_dim),
            dtype=np.float32
        )
        
        if self.current_ast:
            self._encode_ast(self.current_ast, ast_encoding, idx=[0])
        
        # Current cursor position (which node to fill next)
        cursor = len(self.pending_nodes)
        
        # Current depth
        depth = self._get_current_depth()
        
        # Valid action mask
        valid_mask = self._get_valid_action_mask()
        
        return {
            'ast': ast_encoding,
            'cursor': cursor,
            'depth': depth,
            'valid_action_mask': valid_mask
        }
    
    def _encode_ast(
        self,
        node: ASTNode,
        encoding: np.ndarray,
        idx: List[int]
    ):
        """Recursively encode AST into fixed-size array."""
        if idx[0] >= self.config.max_nodes:
            return
        
        i = idx[0]
        idx[0] += 1
        
        # Type one-hot: [operator, operand, constant, param]
        type_map = {'operator': 0, 'operand': 1, 'constant': 2, 'param': 3}
        type_idx = type_map.get(node.node_type, 3)
        encoding[i, type_idx] = 1.0
        
        # Operator ID (if operator)
        if node.node_type == 'operator':
            op_names = list(OPERATORS.keys())
            if node.value.name in op_names:
                op_idx = op_names.index(node.value.name)
                encoding[i, 4 + op_idx] = 1.0
        
        # Dimension type
        dim_offset = 4 + len(OPERATORS)
        dim_idx = list(DimType).index(node.dim_type)
        encoding[i, dim_offset + dim_idx] = 1.0
        
        # Depth (normalized)
        depth_offset = dim_offset + len(DimType)
        encoding[i, depth_offset] = node.depth / self.config.max_depth
        
        # Size (normalized)
        encoding[i, depth_offset + 1] = node.size / self.config.max_nodes
        
        # Is filled flag
        encoding[i, depth_offset + 2] = 1.0
        
        # Recursively encode children
        for child in node.children:
            if child is not None and child.node_type != 'param':
                self._encode_ast(child, encoding, idx)
    
    def _get_info(self) -> Dict:
        """Get info dict for debugging."""
        return {
            'step': self.step_count,
            'valid_actions': self.valid_actions_taken,
            'invalid_actions': self.invalid_actions_taken,
            'pending_nodes': len(self.pending_nodes),
            'expression': self.expression,
            'episode_reward': self.episode_reward,
            'is_complete': self._is_complete()
        }
    
    def render(self):
        """Render current state."""
        if self.render_mode == "human" or self.render_mode == "ansi":
            print(f"\n{'='*60}")
            print(f"Step: {self.step_count}")
            print(f"Expression: {self.expression or '(building...)'}")
            print(f"Pending nodes: {len(self.pending_nodes)}")
            print(f"Episode reward: {self.episode_reward:.4f}")
            print(f"{'='*60}")
    
    def get_expression(self) -> str:
        """Get current expression string."""
        if self.current_ast:
            return build_expression_from_ast(self.current_ast)
        return ""
    
    def get_ast(self) -> Optional[ASTNode]:
        """Get current AST."""
        return self.current_ast
    
    def set_operands(self, operands: List[Operand]):
        """Update available operands and rebuild action space."""
        self.operands = operands
        self.grammar.operands = operands
        self._build_action_space()
    
    def set_alpha_pool(self, pool: List[Dict]):
        """Update alpha pool for diversity checking."""
        self.alpha_pool = pool


def create_env(
    brain_session = None,
    field_list: List[str] = None,
    dataset: str = "model110",
    config: EnvConfig = None,
    **kwargs
) -> AlphaEnv:
    """
    Factory function to create configured AlphaEnv.
    
    Args:
        brain_session: Brain API session
        field_list: List of available field names
        dataset: Dataset name
        config: Environment config
        **kwargs: Additional config overrides
        
    Returns:
        Configured AlphaEnv instance
    """
    config = config or EnvConfig()
    
    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create operands from field list
    operands = []
    if field_list:
        operands = create_default_operands(field_list, dataset)
    
    return AlphaEnv(
        config=config,
        operands=operands,
        brain_session=brain_session
    )
