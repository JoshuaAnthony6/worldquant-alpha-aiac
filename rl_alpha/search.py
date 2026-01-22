"""
MCTS/Beam Search Engine for Alpha Expression Generation

Implements Alpha² style search with:
- Beam search with configurable width
- Policy network guided expansion
- UCB1 selection with diversity bonus
- Dimension-aware pruning

Reference: Alpha² (2024) + AlphaGen (KDD 2023)
"""

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, Callable, Union
from collections import defaultdict
import heapq
import logging
import time

from .grammar import (
    ExpressionGrammar,
    ASTNode,
    Operator,
    Operand,
    DimType,
    OPERATORS,
    build_expression_from_ast
)

logger = logging.getLogger(__name__)


@dataclass
class MCTSStats:
    """Statistics for an MCTS node."""
    visits: int = 0
    total_value: float = 0.0
    mean_value: float = 0.0
    best_value: float = float('-inf')
    best_child: Optional['MCTSNode'] = None
    
    def update(self, value: float):
        self.visits += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visits
        if value > self.best_value:
            self.best_value = value


@dataclass
class MCTSNode:
    """
    Node in the MCTS tree representing a partial or complete expression.
    
    Attributes:
        ast: Current AST at this node
        action: Action that led to this node
        parent: Parent node
        children: Child nodes
        stats: MCTS statistics
        pending_args: Unfilled argument positions
        depth: Tree depth
    """
    ast: Optional[ASTNode]
    action: Optional[Tuple[str, Any]] = None
    parent: Optional['MCTSNode'] = None
    children: Dict[Tuple[str, Any], 'MCTSNode'] = field(default_factory=dict)
    stats: MCTSStats = field(default_factory=MCTSStats)
    pending_args: List[Tuple[ASTNode, int, Set[DimType]]] = field(default_factory=list)
    depth: int = 0
    is_terminal: bool = False
    expression: str = ""
    
    def is_fully_expanded(self, valid_actions: List[Tuple[str, Any]]) -> bool:
        """Check if all valid actions have been tried."""
        return len(self.children) >= len(valid_actions)
    
    def ucb1_score(self, c: float = 1.414) -> float:
        """Calculate UCB1 score for node selection."""
        if self.stats.visits == 0:
            return float('inf')
        
        if self.parent is None:
            return self.stats.mean_value
        
        exploitation = self.stats.mean_value
        exploration = c * math.sqrt(math.log(self.parent.stats.visits) / self.stats.visits)
        
        return exploitation + exploration
    
    def get_best_child(self, c: float = 1.414) -> Optional['MCTSNode']:
        """Get child with highest UCB1 score."""
        if not self.children:
            return None
        
        return max(self.children.values(), key=lambda n: n.ucb1_score(c))


class MCTSEngine:
    """
    Monte Carlo Tree Search engine for alpha expression generation.
    
    Uses policy network to guide expansion and UCB1 for selection.
    """
    
    def __init__(
        self,
        grammar: ExpressionGrammar,
        policy_fn: Optional[Callable] = None,
        evaluate_fn: Optional[Callable] = None,
        c_puct: float = 1.414,
        max_depth: int = 6,
        max_simulations: int = 100,
        diversity_weight: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Initialize MCTS engine.
        
        Args:
            grammar: Expression grammar
            policy_fn: Function to get action probabilities from state
            evaluate_fn: Function to evaluate terminal expressions
            c_puct: UCB exploration constant
            max_depth: Maximum tree depth
            max_simulations: Maximum simulations per search
            diversity_weight: Weight for diversity bonus in selection
            temperature: Temperature for action sampling
        """
        self.grammar = grammar
        self.policy_fn = policy_fn
        self.evaluate_fn = evaluate_fn
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.max_simulations = max_simulations
        self.diversity_weight = diversity_weight
        self.temperature = temperature
        
        # Track seen expressions for diversity
        self.seen_expressions: Set[str] = set()
        self.expression_hashes: Set[str] = set()
    
    def search(
        self,
        initial_state: Optional[ASTNode] = None,
        num_simulations: int = None
    ) -> Tuple[ASTNode, float]:
        """
        Perform MCTS search to find best expression.
        
        Args:
            initial_state: Starting AST (None for new expression)
            num_simulations: Number of simulations (overrides default)
            
        Returns:
            (best_ast, best_value)
        """
        num_simulations = num_simulations or self.max_simulations
        
        # Create root node
        root = MCTSNode(
            ast=initial_state,
            pending_args=[(None, 0, {DimType.ANY})] if initial_state is None else []
        )
        
        for sim in range(num_simulations):
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.is_terminal and node.pending_args:
                node = self._expand(node)
            
            # Simulation/Evaluation
            value = self._evaluate(node)
            
            # Backpropagation
            self._backpropagate(node, value)
        
        # Return best expression found
        best_node = self._get_best_terminal(root)
        if best_node and best_node.ast:
            return best_node.ast, best_node.stats.best_value
        
        return None, float('-inf')
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select leaf node using UCB1."""
        while node.children and not node.is_terminal:
            # Check if fully expanded
            valid_actions = self._get_valid_actions(node)
            
            if not node.is_fully_expanded(valid_actions):
                return node
            
            # Select best child
            best_child = node.get_best_child(self.c_puct)
            if best_child is None:
                break
            node = best_child
        
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a new child."""
        valid_actions = self._get_valid_actions(node)
        
        # Filter out already tried actions
        untried = [a for a in valid_actions if a not in node.children]
        
        if not untried:
            return node
        
        # Use policy to prioritize actions if available
        if self.policy_fn:
            probs = self.policy_fn(node, untried)
            action_idx = np.random.choice(len(untried), p=probs)
            action = untried[action_idx]
        else:
            action = random.choice(untried)
        
        # Create child node
        child_ast, child_pending = self._apply_action(node, action)
        
        child = MCTSNode(
            ast=child_ast,
            action=action,
            parent=node,
            pending_args=child_pending,
            depth=node.depth + 1,
            is_terminal=len(child_pending) == 0
        )
        
        if child.is_terminal and child.ast:
            child.expression = build_expression_from_ast(child.ast)
        
        node.children[action] = child
        return child
    
    def _evaluate(self, node: MCTSNode) -> float:
        """Evaluate a node (terminal or through rollout)."""
        if node.is_terminal:
            if self.evaluate_fn and node.expression:
                # Check if we've seen this expression
                if node.expression in self.seen_expressions:
                    return -0.5  # Penalize duplicates
                
                self.seen_expressions.add(node.expression)
                
                if node.ast:
                    ast_hash = node.ast.hash()
                    if ast_hash in self.expression_hashes:
                        return -0.3  # Penalize structural duplicates
                    self.expression_hashes.add(ast_hash)
                
                return self.evaluate_fn(node.expression)
            return 0.0
        
        # Rollout for non-terminal nodes
        return self._rollout(node)
    
    def _rollout(self, node: MCTSNode) -> float:
        """Perform random rollout from node."""
        # Create a copy of pending args
        pending = list(node.pending_args)
        current_ast = node.ast
        depth = node.depth
        
        while pending and depth < self.max_depth:
            parent, arg_idx, required_dim = pending.pop()
            
            # Get valid actions for this position
            valid_actions = self.grammar.get_valid_actions(
                parent, depth, 
                current_ast.size if current_ast else 0,
                required_dim
            )
            
            if not valid_actions:
                return -1.0  # Dead end
            
            # Random action
            action = random.choice(valid_actions)
            action_type, action_value = action
            
            if action_type == 'operand':
                # Create leaf node
                leaf = ASTNode(
                    node_type='operand',
                    value=action_value,
                    dim_type=action_value.dim_type
                )
                if parent and parent.node_type == 'operator':
                    parent.children[arg_idx] = leaf
                else:
                    current_ast = leaf
            
            elif action_type == 'operator':
                op, params = action_value
                # Create operator node
                children = [None] * op.arity
                param_children = [
                    ASTNode(node_type='param', value=f"{k}={v}")
                    for k, v in params.items()
                ]
                
                op_node = ASTNode(
                    node_type='operator',
                    value=op,
                    children=children + param_children
                )
                
                if parent and parent.node_type == 'operator':
                    parent.children[arg_idx] = op_node
                else:
                    current_ast = op_node
                
                # Add children to pending
                for i in range(op.arity - 1, -1, -1):
                    req = op.input_types[i] if i < len(op.input_types) else {DimType.ANY}
                    pending.append((op_node, i, req))
            
            depth += 1
        
        # Evaluate rollout result
        if current_ast and not pending:
            expr = build_expression_from_ast(current_ast)
            if expr and self.evaluate_fn:
                return self.evaluate_fn(expr) * 0.5  # Discount rollout value
            return 0.1  # Small positive for complete expression
        
        return -0.1  # Incomplete
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        while node is not None:
            node.stats.update(value)
            node = node.parent
    
    def _get_valid_actions(self, node: MCTSNode) -> List[Tuple[str, Any]]:
        """Get valid actions for a node."""
        if not node.pending_args:
            return []
        
        parent, arg_idx, required_dim = node.pending_args[-1]
        current_depth = node.depth
        current_size = node.ast.size if node.ast else 0
        
        return self.grammar.get_valid_actions(
            parent, current_depth, current_size, required_dim
        )
    
    def _apply_action(
        self,
        node: MCTSNode,
        action: Tuple[str, Any]
    ) -> Tuple[Optional[ASTNode], List[Tuple]]:
        """Apply action to create new state."""
        action_type, action_value = action
        
        # Copy pending args
        pending = list(node.pending_args)
        if not pending:
            return node.ast, []
        
        parent, arg_idx, required_dim = pending.pop()
        
        # Copy AST (shallow copy for now)
        new_ast = node.ast
        
        if action_type == 'operand':
            operand = action_value
            leaf = ASTNode(
                node_type='operand',
                value=operand,
                dim_type=operand.dim_type
            )
            
            if parent and parent.node_type == 'operator':
                parent.children[arg_idx] = leaf
            else:
                new_ast = leaf
        
        elif action_type == 'operator':
            op, params = action_value
            if isinstance(params, tuple):
                params = dict(params)
            
            children = [None] * op.arity
            param_children = [
                ASTNode(node_type='param', value=f"{k}={v}")
                for k, v in params.items()
            ]
            
            op_node = ASTNode(
                node_type='operator',
                value=op,
                children=children + param_children
            )
            
            if parent and parent.node_type == 'operator':
                parent.children[arg_idx] = op_node
            else:
                new_ast = op_node
            
            # Add new pending args
            for i in range(op.arity - 1, -1, -1):
                req = op.input_types[i] if i < len(op.input_types) else {DimType.ANY}
                pending.append((op_node, i, req))
        
        return new_ast, pending
    
    def _get_best_terminal(self, root: MCTSNode) -> Optional[MCTSNode]:
        """Get best terminal node from tree."""
        best = None
        best_value = float('-inf')
        
        def traverse(node):
            nonlocal best, best_value
            if node.is_terminal and node.stats.best_value > best_value:
                best = node
                best_value = node.stats.best_value
            for child in node.children.values():
                traverse(child)
        
        traverse(root)
        return best


class BeamSearchEngine:
    """
    Beam search engine for alpha expression generation.
    
    Maintains top-K candidates at each depth level.
    More efficient than MCTS for single-pass generation.
    """
    
    def __init__(
        self,
        grammar: ExpressionGrammar,
        beam_width: int = 50,
        max_depth: int = 6,
        policy_fn: Optional[Callable] = None,
        evaluate_fn: Optional[Callable] = None,
        diversity_weight: float = 0.2,
        prune_threshold: float = -0.5
    ):
        """
        Initialize beam search engine.
        
        Args:
            grammar: Expression grammar
            beam_width: Number of candidates to keep per level
            max_depth: Maximum expression depth
            policy_fn: Function to score candidate actions
            evaluate_fn: Function to evaluate complete expressions
            diversity_weight: Weight for diversity in beam selection
            prune_threshold: Minimum score to keep candidate
        """
        self.grammar = grammar
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.policy_fn = policy_fn
        self.evaluate_fn = evaluate_fn
        self.diversity_weight = diversity_weight
        self.prune_threshold = prune_threshold
        
        # Diversity tracking
        self.seen_hashes: Set[str] = set()
        
        # Template patterns optimized for CONSULTANT CRITERIA
        # Target: Sharpe>1.58, Fitness>1.0, Margin>15bp, Turnover 5-20%
        # Focus on SIMPLE expressions: ≤6 operators, ≤3 fields
        self.templates = [
            # ===== TIER 1: Simplest (1-2 ops) - Best for Sharpe =====
            "rank({field})",
            "group_rank({field}, sector)",
            "group_rank({field}, subindustry)",
            "ts_rank({field}, {window})",
            "ts_zscore({field}, {window})",
            "ts_ir({field}, {window})",
            "-rank({field})",  # Sign flip can improve
            "-group_rank({field}, sector)",
            
            # ===== TIER 2: Quality factor combos (proven high Sharpe) =====
            "rank(group_rank({field}, sector))",
            "rank(group_rank({field}, subindustry))",
            "scale(rank({field}))",
            "scale(group_rank({field}, sector))",
            
            # ===== TIER 3: Time-series momentum (good turnover) =====
            "rank(ts_delta({field}, {window}))",
            "rank(ts_zscore({field}, {window}))",
            "ts_decay_linear(rank({field}), {window})",
            "ts_mean(rank({field}), {window})",
            
            # ===== TIER 4: Proven best patterns =====
            "ts_mean(rank(ts_zscore({field}, {long})), {short})",
            "ts_zscore(rank(group_rank({field}, sector)), {window})",
            "rank(ts_ir({field}, {window}))",
            
            # ===== TIER 5: Cross-sectional + time-series combos =====
            "ts_delta(group_rank({field}, sector), {window})",
            "ts_rank(group_rank({field}, subindustry), {window})",
            "group_rank(ts_zscore({field}, {window}), sector)",
        ]
        # Window options - shorter windows = higher turnover
        self.windows = [5, 10, 20, 40, 60]
    
    def _generate_template_candidates(
        self, 
        num_templates: int = 20,
        iteration_offset: int = 0
    ) -> List[Tuple[None, str, float]]:
        """Generate diverse candidates from templates using systematic variation."""
        import random
        candidates = []
        fields = [op.name for op in self.grammar.operands]
        
        # Use iteration offset to select different templates/fields
        num_templates_avail = len(self.templates)
        num_fields = len(fields)
        num_windows = len(self.windows)
        
        for i in range(num_templates):
            # Systematic coverage with iteration-based offset
            template_idx = (i + iteration_offset * 3) % num_templates_avail
            field_idx = (i + iteration_offset * 5) % num_fields
            window_idx = (i + iteration_offset * 7) % num_windows
            
            template = self.templates[template_idx]
            field = fields[field_idx]
            window = self.windows[window_idx]
            long_window = [60, 120, 252][(i + iteration_offset) % 3]
            short_window = [5, 10, 20][(i + iteration_offset * 2) % 3]
            
            expr = template.format(
                field=field, 
                window=window,
                long=long_window,
                short=short_window
            )
            
            # Skip duplicates
            if expr in self.seen_hashes:
                continue
            self.seen_hashes.add(expr)
            
            # Score with proxy
            if self.evaluate_fn:
                score = self.evaluate_fn(expr)
            else:
                score = 0.5
            
            candidates.append((None, expr, score))
        
        return candidates
    
    def search(
        self,
        num_candidates: int = 100,
        reset_diversity: bool = True,
        include_templates: bool = True,
        iteration_seed: int = 0
    ) -> List[Tuple[ASTNode, str, float]]:
        """
        Perform beam search to generate candidate expressions.
        
        Args:
            num_candidates: Number of final candidates to return
            reset_diversity: Whether to clear seen_hashes for fresh search
            include_templates: Whether to include template-based candidates
            iteration_seed: Seed offset for randomization (use iteration number)
            
        Returns:
            List of (ast, expression, score) tuples
        """
        import random
        random.seed(42 + iteration_seed)  # Deterministic but varies per iteration
        
        # Reset diversity tracking for fresh search
        if reset_diversity:
            self.seen_hashes.clear()
        
        # Start with template-based candidates for diversity
        completed = []
        if include_templates:
            template_candidates = self._generate_template_candidates(
                num_templates=num_candidates // 2,
                iteration_offset=iteration_seed
            )
            completed.extend(template_candidates)
        
        # Initialize beam with empty state
        beam: List[Tuple[float, ASTNode, List[Tuple]]] = [
            (0.0, None, [(None, 0, {DimType.ANY})])  # (score, ast, pending)
        ]
        
        for depth in range(self.max_depth * 2):  # Allow for operator arity
            if not beam:
                break
            
            next_beam = []
            
            for score, ast, pending in beam:
                if not pending:
                    # Complete expression
                    if ast:
                        expr = build_expression_from_ast(ast)
                        if expr:
                            # Evaluate final expression
                            if self.evaluate_fn:
                                final_score = self.evaluate_fn(expr)
                            else:
                                final_score = score
                            
                            # Check diversity
                            ast_hash = ast.hash()
                            if ast_hash not in self.seen_hashes:
                                self.seen_hashes.add(ast_hash)
                                completed.append((ast, expr, final_score))
                    continue
                
                # Expand candidate
                parent, arg_idx, required_dim = pending[-1]
                new_pending = pending[:-1]
                
                current_size = ast.size if ast else 0
                current_depth = depth // 2
                valid_actions = self.grammar.get_valid_actions(
                    parent, current_depth, current_size, required_dim
                )
                
                # Get parent operator name for anti-nesting
                parent_op_name = None
                if parent and parent.node_type == 'operator':
                    parent_op_name = parent.value.name
                
                # Score and rank actions with anti-nesting and depth penalties
                action_scores = []
                for action in valid_actions:
                    action_type, action_value = action
                    
                    if self.policy_fn:
                        base_score = self.policy_fn(ast, [action])[0]
                    else:
                        base_score = 1.0
                    
                    # Apply depth-based preference for operands (leaves)
                    if action_type == 'operand':
                        # Bonus for operands at deeper levels
                        depth_bonus = 0.3 * (current_depth / self.max_depth)
                        base_score += depth_bonus
                    elif action_type == 'operator':
                        op, params = action_value
                        # Penalty for consecutive same operator (anti-nesting)
                        if parent_op_name and op.name == parent_op_name:
                            base_score *= 0.1  # Strong penalty for repeating
                        # Penalty for unary operators at deep levels
                        if op.arity == 1 and current_depth >= 3:
                            base_score *= 0.5
                        # Penalty for group_* operators at any level > 1
                        if 'group' in op.name and current_depth >= 2:
                            base_score *= 0.3
                    
                    action_scores.append(base_score)
                
                # Take top actions
                top_k = min(self.beam_width // max(1, len(beam)), len(valid_actions))
                top_indices = np.argsort(action_scores)[-top_k:][::-1]
                
                for idx in top_indices:
                    action = valid_actions[idx]
                    action_type, action_value = action
                    new_score = score + action_scores[idx] * 0.1
                    
                    if action_type == 'operand':
                        operand = action_value
                        leaf = ASTNode(
                            node_type='operand',
                            value=operand,
                            dim_type=operand.dim_type
                        )
                        
                        if parent and parent.node_type == 'operator':
                            # Deep clone tree and fill the slot, update pending refs
                            new_ast, updated_pending = self._fill_slot(
                                ast, parent, arg_idx, leaf, new_pending
                            )
                        else:
                            new_ast = leaf
                            updated_pending = new_pending.copy()
                        
                        next_beam.append((new_score, new_ast, updated_pending))
                    
                    elif action_type == 'operator':
                        op, params = action_value
                        if isinstance(params, tuple):
                            params = dict(params)
                        
                        children = [None] * op.arity
                        param_children = [
                            ASTNode(node_type='param', value=f"{k}={v}")
                            for k, v in params.items()
                        ]
                        
                        op_node = ASTNode(
                            node_type='operator',
                            value=op,
                            children=children + param_children
                        )
                        
                        if parent and parent.node_type == 'operator':
                            # Deep clone tree and fill the slot, update pending refs
                            new_ast, updated_pending = self._fill_slot(
                                ast, parent, arg_idx, op_node, new_pending
                            )
                            # op_node was inserted directly (not cloned), so it's the actual node
                            actual_op_node = op_node
                        else:
                            new_ast = op_node
                            actual_op_node = op_node
                            updated_pending = new_pending.copy()
                        
                        # Add children to pending (reference the actual node in tree)
                        child_pending = updated_pending
                        for i in range(op.arity - 1, -1, -1):
                            req = op.input_types[i] if i < len(op.input_types) else {DimType.ANY}
                            child_pending.append((actual_op_node, i, req))
                        
                        next_beam.append((new_score, new_ast, child_pending))
            
            # Select top candidates for next iteration
            if next_beam:
                next_beam.sort(key=lambda x: x[0], reverse=True)
                
                # Apply diversity filtering
                diverse_beam = []
                seen_structures = set()
                
                for item in next_beam:
                    score, ast, pending = item
                    if ast:
                        struct = self._get_structure_signature(ast)
                        if struct not in seen_structures or len(diverse_beam) < self.beam_width // 2:
                            seen_structures.add(struct)
                            diverse_beam.append(item)
                    else:
                        diverse_beam.append(item)
                    
                    if len(diverse_beam) >= self.beam_width:
                        break
                
                beam = diverse_beam
        
        # Sort completed by score
        completed.sort(key=lambda x: x[2], reverse=True)
        
        return completed[:num_candidates]
    
    def _deep_clone_ast(self, node: ASTNode, node_map: Dict = None) -> Tuple[ASTNode, Dict]:
        """
        Deep clone an AST, returning clone and mapping from old to new nodes.
        """
        if node is None:
            return None, node_map or {}
        
        if node_map is None:
            node_map = {}
        
        # Clone children first
        new_children = []
        for child in node.children:
            if child is None:
                new_children.append(None)
            else:
                child_clone, node_map = self._deep_clone_ast(child, node_map)
                new_children.append(child_clone)
        
        # Create clone of this node
        new_node = ASTNode(
            node_type=node.node_type,
            value=node.value,
            children=new_children,
            dim_type=node.dim_type
        )
        node_map[id(node)] = new_node
        
        return new_node, node_map
    
    def _fill_slot(
        self, 
        root: ASTNode, 
        target_node: ASTNode, 
        child_idx: int, 
        fill_value: ASTNode,
        pending: List[Tuple] = None
    ) -> Tuple[ASTNode, List[Tuple]]:
        """
        Fill a slot in the AST by deep cloning and modifying.
        
        Returns (new_root, updated_pending) with slot filled and pending refs updated.
        """
        if root is None:
            return fill_value, pending or []
        
        # Deep clone the tree
        new_root, node_map = self._deep_clone_ast(root)
        
        # Find the cloned target node
        cloned_target = node_map.get(id(target_node))
        if cloned_target is not None:
            cloned_target.children[child_idx] = fill_value
        
        # Update pending references to use cloned nodes
        updated_pending = []
        if pending:
            for old_parent, idx, req_dim in pending:
                new_parent = node_map.get(id(old_parent), old_parent)
                updated_pending.append((new_parent, idx, req_dim))
        
        return new_root, updated_pending
    
    def _get_structure_signature(self, ast: ASTNode) -> str:
        """Get structural signature for diversity checking."""
        if ast.node_type == 'operand':
            return 'O'
        elif ast.node_type == 'operator':
            children_sig = ''.join(
                self._get_structure_signature(c) 
                for c in ast.children if c and c.node_type != 'param'
            )
            return f"{ast.value.name}({children_sig})"
        return ''


class HybridSearchEngine:
    """
    Hybrid search combining beam search for efficiency 
    and MCTS for refinement.
    """
    
    def __init__(
        self,
        grammar: ExpressionGrammar,
        beam_width: int = 50,
        mcts_simulations: int = 50,
        max_depth: int = 6,
        policy_fn: Optional[Callable] = None,
        evaluate_fn: Optional[Callable] = None
    ):
        """
        Initialize hybrid search engine.
        
        Args:
            grammar: Expression grammar
            beam_width: Beam search width
            mcts_simulations: MCTS simulations for refinement
            max_depth: Maximum expression depth
            policy_fn: Policy scoring function
            evaluate_fn: Expression evaluation function
        """
        self.beam_engine = BeamSearchEngine(
            grammar=grammar,
            beam_width=beam_width,
            max_depth=max_depth,
            policy_fn=policy_fn,
            evaluate_fn=evaluate_fn
        )
        
        self.mcts_engine = MCTSEngine(
            grammar=grammar,
            policy_fn=policy_fn,
            evaluate_fn=evaluate_fn,
            max_depth=max_depth,
            max_simulations=mcts_simulations
        )
    
    def search(
        self,
        num_candidates: int = 100,
        refine_top_k: int = 10
    ) -> List[Tuple[ASTNode, str, float]]:
        """
        Perform hybrid search.
        
        1. Use beam search for broad exploration
        2. Use MCTS to refine top candidates
        
        Args:
            num_candidates: Number of candidates from beam search
            refine_top_k: Number of top candidates to refine with MCTS
            
        Returns:
            List of (ast, expression, score) tuples
        """
        logger.info(f"Starting beam search with width {self.beam_engine.beam_width}")
        
        # Phase 1: Beam search
        beam_results = self.beam_engine.search(num_candidates=num_candidates)
        
        logger.info(f"Beam search found {len(beam_results)} candidates")
        
        # Phase 2: MCTS refinement of top candidates
        refined_results = []
        
        for ast, expr, score in beam_results[:refine_top_k]:
            logger.debug(f"Refining: {expr[:50]}... (score: {score:.4f})")
            
            # Use MCTS starting from this expression's structure
            refined_ast, refined_score = self.mcts_engine.search(
                initial_state=ast,
                num_simulations=self.mcts_engine.max_simulations // 2
            )
            
            if refined_ast and refined_score > score:
                refined_expr = build_expression_from_ast(refined_ast)
                refined_results.append((refined_ast, refined_expr, refined_score))
            else:
                refined_results.append((ast, expr, score))
        
        # Add remaining beam results
        refined_results.extend(beam_results[refine_top_k:])
        
        # Sort by score
        refined_results.sort(key=lambda x: x[2], reverse=True)
        
        return refined_results


def create_search_engine(
    grammar: ExpressionGrammar,
    method: str = "beam",
    **kwargs
) -> Union[BeamSearchEngine, MCTSEngine, HybridSearchEngine]:
    """
    Factory function to create search engine.
    
    Args:
        grammar: Expression grammar
        method: Search method ("beam", "mcts", "hybrid")
        **kwargs: Method-specific arguments
        
    Returns:
        Configured search engine
    """
    if method == "beam":
        return BeamSearchEngine(grammar=grammar, **kwargs)
    elif method == "mcts":
        return MCTSEngine(grammar=grammar, **kwargs)
    elif method == "hybrid":
        return HybridSearchEngine(grammar=grammar, **kwargs)
    else:
        raise ValueError(f"Unknown search method: {method}")
