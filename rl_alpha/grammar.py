"""
DSL Grammar + Dimension System for Alpha Expression Generation

Implements Alpha² style dimension-aware grammar that:
1. Defines typed operators with input/output dimensions
2. Pre-computes dimension compatibility matrix
3. Prunes invalid operator combinations at expansion time

Reference: Alpha² (2024) - Dimension consistency constraints
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Set, Tuple, Any, Union
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class DimType(Enum):
    """Dimension types for operands and operators."""
    PRICE = auto()      # Price-like values (close, open, high, low)
    VOLUME = auto()     # Volume/quantity values
    RATIO = auto()      # Dimensionless ratios (returns, growth rates)
    COUNT = auto()      # Integer counts
    RANK = auto()       # Cross-sectional ranks (0-1)
    ZSCORE = auto()     # Standardized values
    BOOLEAN = auto()    # True/False values
    ANY = auto()        # Accepts any dimension
    UNKNOWN = auto()    # Unknown/untyped


# Dimension compatibility matrix for operations
# (left_dim, right_dim) -> result_dim for binary operations
DIMENSION_COMPAT = {
    # Addition/Subtraction - only same dimensions
    (DimType.PRICE, DimType.PRICE): DimType.PRICE,
    (DimType.VOLUME, DimType.VOLUME): DimType.VOLUME,
    (DimType.RATIO, DimType.RATIO): DimType.RATIO,
    (DimType.RANK, DimType.RANK): DimType.RATIO,  # rank diff -> ratio
    (DimType.ZSCORE, DimType.ZSCORE): DimType.ZSCORE,
    
    # Multiplication - produces new dimensions
    (DimType.PRICE, DimType.RATIO): DimType.PRICE,
    (DimType.RATIO, DimType.PRICE): DimType.PRICE,
    (DimType.VOLUME, DimType.RATIO): DimType.VOLUME,
    (DimType.RATIO, DimType.VOLUME): DimType.VOLUME,
    (DimType.RATIO, DimType.RATIO): DimType.RATIO,
    (DimType.RANK, DimType.RATIO): DimType.RATIO,
    (DimType.RATIO, DimType.RANK): DimType.RATIO,
    (DimType.RANK, DimType.RANK): DimType.RATIO,
    
    # Division - produces ratios
    (DimType.PRICE, DimType.PRICE): DimType.RATIO,
    (DimType.VOLUME, DimType.VOLUME): DimType.RATIO,
}


@dataclass
class Operator:
    """
    Definition of an operator in the alpha DSL.
    
    Attributes:
        name: Operator name (e.g., "ts_delta", "rank")
        input_types: List of accepted input dimension types
        output_type: Output dimension type (or callable for dynamic)
        arity: Number of arguments
        category: Operator category for organization
        params: Additional parameter specs (e.g., window)
        syntax: Syntax template for expression generation
    """
    name: str
    input_types: List[Set[DimType]]  # Per-argument accepted types
    output_type: Union[DimType, str]  # 'same' means same as input
    arity: int
    category: str = "general"
    params: Dict[str, Any] = field(default_factory=dict)
    syntax: str = ""
    
    def __post_init__(self):
        if not self.syntax:
            if self.arity == 1:
                self.syntax = f"{self.name}({{0}})"
            elif self.arity == 2:
                self.syntax = f"{self.name}({{0}}, {{1}})"
            else:
                args = ", ".join(f"{{{i}}}" for i in range(self.arity))
                self.syntax = f"{self.name}({args})"
    
    def get_output_dim(self, input_dims: List[DimType]) -> DimType:
        """Determine output dimension based on input dimensions."""
        if isinstance(self.output_type, DimType):
            return self.output_type
        elif self.output_type == 'same':
            return input_dims[0] if input_dims else DimType.UNKNOWN
        elif self.output_type == 'ratio':
            return DimType.RATIO
        elif self.output_type == 'rank':
            return DimType.RANK
        elif self.output_type == 'zscore':
            return DimType.ZSCORE
        return DimType.UNKNOWN
    
    def accepts_input(self, arg_idx: int, dim: DimType) -> bool:
        """Check if operator accepts dimension for given argument."""
        if arg_idx >= len(self.input_types):
            return False
        accepted = self.input_types[arg_idx]
        return DimType.ANY in accepted or dim in accepted or dim == DimType.ANY


@dataclass
class Operand:
    """
    A data field operand in the alpha expression.
    
    Attributes:
        name: Field name (e.g., "mdl110_growth", "close")
        dim_type: Dimension type
        dataset: Source dataset
    """
    name: str
    dim_type: DimType = DimType.RATIO
    dataset: str = "model110"
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Operand):
            return self.name == other.name
        return False


@dataclass 
class ASTNode:
    """
    Abstract Syntax Tree node for alpha expressions.
    
    Can represent either:
    - An operator node with children
    - A leaf operand node
    - A constant value node
    """
    node_type: str  # 'operator', 'operand', 'constant', 'param'
    value: Union[Operator, Operand, float, int, str]
    children: List['ASTNode'] = field(default_factory=list)
    dim_type: DimType = DimType.UNKNOWN
    
    def __post_init__(self):
        # Infer dimension type
        if self.node_type == 'operand' and isinstance(self.value, Operand):
            self.dim_type = self.value.dim_type
        elif self.node_type == 'constant':
            self.dim_type = DimType.RATIO
        elif self.node_type == 'operator' and isinstance(self.value, Operator):
            child_dims = [c.dim_type for c in self.children if c is not None and c.node_type != 'param']
            self.dim_type = self.value.get_output_dim(child_dims)
    
    def to_dict(self) -> Dict:
        """Convert AST to dictionary for serialization."""
        if self.node_type == 'operator':
            return {
                'type': 'operator',
                'name': self.value.name,
                'children': [c.to_dict() for c in self.children if c is not None],
                'dim': self.dim_type.name
            }
        elif self.node_type == 'operand':
            return {
                'type': 'operand',
                'name': self.value.name,
                'dim': self.dim_type.name
            }
        elif self.node_type == 'constant':
            return {
                'type': 'constant',
                'value': self.value,
                'dim': self.dim_type.name
            }
        else:
            return {
                'type': self.node_type,
                'value': str(self.value)
            }
    
    def hash(self) -> str:
        """Compute structural hash for AST deduplication."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    @property
    def depth(self) -> int:
        """Calculate tree depth."""
        valid_children = [c for c in self.children if c is not None]
        if not valid_children:
            return 1
        return 1 + max(c.depth for c in valid_children)
    
    @property
    def size(self) -> int:
        """Calculate total number of nodes."""
        return 1 + sum(c.size for c in self.children if c is not None)


# Define all Brain operators with dimension types
OPERATORS: Dict[str, Operator] = {}

def _register_operators():
    """Register all Brain operators with their dimension specifications."""
    global OPERATORS
    
    # === Arithmetic Operators ===
    OPERATORS['add'] = Operator(
        name='add',
        input_types=[{DimType.ANY}, {DimType.ANY}],
        output_type='same',
        arity=2,
        category='arithmetic'
    )
    
    OPERATORS['subtract'] = Operator(
        name='subtract',
        input_types=[{DimType.ANY}, {DimType.ANY}],
        output_type='same',
        arity=2,
        category='arithmetic'
    )
    
    OPERATORS['multiply'] = Operator(
        name='multiply',
        input_types=[{DimType.ANY}, {DimType.ANY}],
        output_type=DimType.RATIO,
        arity=2,
        category='arithmetic'
    )
    
    OPERATORS['divide'] = Operator(
        name='divide',
        input_types=[{DimType.ANY}, {DimType.ANY}],
        output_type=DimType.RATIO,
        arity=2,
        category='arithmetic'
    )
    
    OPERATORS['abs'] = Operator(
        name='abs',
        input_types=[{DimType.ANY}],
        output_type='same',
        arity=1,
        category='arithmetic'
    )
    
    OPERATORS['log'] = Operator(
        name='log',
        input_types=[{DimType.PRICE, DimType.VOLUME, DimType.RATIO}],
        output_type=DimType.RATIO,
        arity=1,
        category='arithmetic'
    )
    
    OPERATORS['power'] = Operator(
        name='power',
        input_types=[{DimType.ANY}, {DimType.RATIO}],
        output_type='same',
        arity=2,
        category='arithmetic',
        syntax='power({0}, {1})'
    )
    
    OPERATORS['signed_power'] = Operator(
        name='signed_power',
        input_types=[{DimType.ANY}, {DimType.RATIO}],
        output_type='same',
        arity=2,
        category='arithmetic'
    )
    
    OPERATORS['inverse'] = Operator(
        name='inverse',
        input_types=[{DimType.ANY}],
        output_type=DimType.RATIO,
        arity=1,
        category='arithmetic'
    )
    
    # === Cross-sectional Operators ===
    OPERATORS['rank'] = Operator(
        name='rank',
        input_types=[{DimType.ANY}],
        output_type=DimType.RANK,
        arity=1,
        category='cross_sectional'
    )
    
    OPERATORS['scale'] = Operator(
        name='scale',
        input_types=[{DimType.ANY}],
        output_type=DimType.RATIO,
        arity=1,
        category='cross_sectional'
    )
    
    OPERATORS['zscore'] = Operator(
        name='zscore',
        input_types=[{DimType.ANY}],
        output_type=DimType.ZSCORE,
        arity=1,
        category='cross_sectional'
    )
    
    OPERATORS['winsorize'] = Operator(
        name='winsorize',
        input_types=[{DimType.ANY}],
        output_type='same',
        arity=1,
        category='cross_sectional',
        params={'std': [3, 4, 5]},
        syntax='winsorize({0}, std={std})'
    )
    
    OPERATORS['quantile'] = Operator(
        name='quantile',
        input_types=[{DimType.ANY}],
        output_type=DimType.RANK,
        arity=1,
        category='cross_sectional'
    )
    
    # === Time Series Operators ===
    OPERATORS['ts_delta'] = Operator(
        name='ts_delta',
        input_types=[{DimType.ANY}],
        output_type='same',
        arity=1,
        category='time_series',
        params={'window': [5, 10, 20, 60, 120, 252]},
        syntax='ts_delta({0}, {window})'
    )
    
    OPERATORS['ts_mean'] = Operator(
        name='ts_mean',
        input_types=[{DimType.ANY}],
        output_type='same',
        arity=1,
        category='time_series',
        params={'window': [5, 10, 20, 60, 120, 252]},
        syntax='ts_mean({0}, {window})'
    )
    
    OPERATORS['ts_std_dev'] = Operator(
        name='ts_std_dev',
        input_types=[{DimType.ANY}],
        output_type=DimType.RATIO,
        arity=1,
        category='time_series',
        params={'window': [10, 20, 60, 120, 252]},
        syntax='ts_std_dev({0}, {window})'
    )
    
    OPERATORS['ts_zscore'] = Operator(
        name='ts_zscore',
        input_types=[{DimType.ANY}],
        output_type=DimType.ZSCORE,
        arity=1,
        category='time_series',
        params={'window': [20, 60, 120, 252]},
        syntax='ts_zscore({0}, {window})'
    )
    
    OPERATORS['ts_rank'] = Operator(
        name='ts_rank',
        input_types=[{DimType.ANY}],
        output_type=DimType.RANK,
        arity=1,
        category='time_series',
        params={'window': [20, 60, 120, 252]},
        syntax='ts_rank({0}, {window}, constant=0)'
    )
    
    OPERATORS['ts_sum'] = Operator(
        name='ts_sum',
        input_types=[{DimType.ANY}],
        output_type='same',
        arity=1,
        category='time_series',
        params={'window': [5, 10, 20, 60]},
        syntax='ts_sum({0}, {window})'
    )
    
    # NOTE: ts_min and ts_max are NOT available in model110, commented out
    # OPERATORS['ts_min'] = Operator(...)
    # OPERATORS['ts_max'] = Operator(...)
    
    OPERATORS['ts_decay_linear'] = Operator(
        name='ts_decay_linear',
        input_types=[{DimType.ANY}],
        output_type='same',
        arity=1,
        category='time_series',
        params={'window': [5, 10, 20, 60]},
        syntax='ts_decay_linear({0}, {window})'
    )
    
    OPERATORS['ts_ir'] = Operator(
        name='ts_ir',
        input_types=[{DimType.ANY}],
        output_type=DimType.RATIO,
        arity=1,
        category='time_series',
        params={'window': [20, 60, 120, 252]},
        syntax='ts_ir({0}, {window})'
    )
    
    OPERATORS['ts_returns'] = Operator(
        name='ts_returns',
        input_types=[{DimType.PRICE, DimType.RATIO}],
        output_type=DimType.RATIO,
        arity=1,
        category='time_series',
        params={'window': [1, 5, 10, 20]},
        syntax='ts_returns({0}, {window}, mode=1)'
    )
    
    # NOTE: ts_argmax and ts_argmin are NOT available in model110, commented out
    # OPERATORS['ts_argmax'] = Operator(...)
    # OPERATORS['ts_argmin'] = Operator(...)
    
    # === Conditional Operators ===
    OPERATORS['if_else'] = Operator(
        name='if_else',
        input_types=[{DimType.BOOLEAN}, {DimType.ANY}, {DimType.ANY}],
        output_type='same',  # same as true/false branch
        arity=3,
        category='conditional',
        syntax='if_else({0}, {1}, {2})'
    )
    
    OPERATORS['greater'] = Operator(
        name='greater',
        input_types=[{DimType.ANY}, {DimType.ANY}],
        output_type=DimType.BOOLEAN,
        arity=2,
        category='conditional'
    )
    
    OPERATORS['less'] = Operator(
        name='less',
        input_types=[{DimType.ANY}, {DimType.ANY}],
        output_type=DimType.BOOLEAN,
        arity=2,
        category='conditional'
    )
    
    # === Group Operators ===
    OPERATORS['group_neutralize'] = Operator(
        name='group_neutralize',
        input_types=[{DimType.ANY}],
        output_type='same',
        arity=1,
        category='group',
        params={'group': ['sector', 'industry', 'subindustry']},
        syntax='group_neutralize({0}, {group})'
    )
    
    OPERATORS['group_rank'] = Operator(
        name='group_rank',
        input_types=[{DimType.ANY}],
        output_type=DimType.RANK,
        arity=1,
        category='group',
        params={'group': ['sector', 'industry', 'subindustry']},
        syntax='group_rank({0}, {group})'
    )

_register_operators()


class ExpressionGrammar:
    """
    Grammar-based expression generation with dimension constraints.
    
    Implements Alpha² style dimension-aware generation that prunes
    invalid combinations during tree expansion.
    """
    
    # Constraints for expression generation
    MAX_DEPTH = 6
    MAX_NODES = 12
    MAX_WINDOWS = 3  # Max number of window parameters
    
    def __init__(
        self,
        operators: Dict[str, Operator] = None,
        operands: List[Operand] = None,
        max_depth: int = MAX_DEPTH,
        max_nodes: int = MAX_NODES
    ):
        """
        Initialize grammar with operators and operands.
        
        Args:
            operators: Dict of available operators
            operands: List of available data field operands
            max_depth: Maximum AST depth
            max_nodes: Maximum number of nodes in AST
        """
        self.operators = operators or OPERATORS
        self.operands = operands or []
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        
        # Pre-compute valid operator pairs for efficiency
        self._valid_pairs = self._compute_valid_pairs()
        
        # Categorize operators
        self.unary_ops = {k: v for k, v in self.operators.items() if v.arity == 1}
        self.binary_ops = {k: v for k, v in self.operators.items() if v.arity == 2}
        self.ternary_ops = {k: v for k, v in self.operators.items() if v.arity == 3}
    
    def _compute_valid_pairs(self) -> Dict[Tuple[str, DimType], List[str]]:
        """
        Pre-compute which operators can follow which, given output dimension.
        
        Returns:
            Dict mapping (parent_op_name, output_dim) -> list of valid child ops
        """
        valid = {}
        for parent_name, parent_op in self.operators.items():
            for dim in DimType:
                key = (parent_name, dim)
                valid[key] = []
                for child_name, child_op in self.operators.items():
                    if child_op.accepts_input(0, dim):
                        valid[key].append(child_name)
        return valid
    
    def get_valid_actions(
        self,
        current_node: Optional[ASTNode],
        current_depth: int,
        current_size: int,
        required_dim: Set[DimType] = None
    ) -> List[Tuple[str, Any]]:
        """
        Get valid actions (operators/operands) for expanding current position.
        
        Args:
            current_node: Current partial AST (or None at root)
            current_depth: Current tree depth
            current_size: Current number of nodes
            required_dim: Required output dimension for this position
            
        Returns:
            List of (action_type, action_value) tuples
        """
        actions = []
        required_dim = required_dim or {DimType.ANY}
        
        # Check depth/size constraints
        can_add_operator = (current_depth < self.max_depth and 
                          current_size < self.max_nodes - 1)
        
        # Add operands (leaf nodes)
        for operand in self.operands:
            if DimType.ANY in required_dim or operand.dim_type in required_dim:
                actions.append(('operand', operand))
        
        # Add operators if depth allows
        if can_add_operator:
            for op_name, op in self.operators.items():
                # Check if output dimension is compatible
                # We need to consider all possible output dimensions
                compatible = False
                if DimType.ANY in required_dim:
                    compatible = True
                elif op.output_type == 'same':
                    # Output depends on input, so could be compatible
                    compatible = True
                elif isinstance(op.output_type, DimType):
                    compatible = op.output_type in required_dim
                elif op.output_type in ['ratio', 'rank', 'zscore']:
                    type_map = {'ratio': DimType.RATIO, 'rank': DimType.RANK, 
                               'zscore': DimType.ZSCORE}
                    compatible = type_map.get(op.output_type, DimType.UNKNOWN) in required_dim
                
                if compatible:
                    # For operators with parameters, add all variants
                    if op.params:
                        param_combos = self._get_param_combinations(op.params)
                        for combo in param_combos:
                            actions.append(('operator', (op, combo)))
                    else:
                        actions.append(('operator', (op, {})))
        
        return actions
    
    def _get_param_combinations(
        self,
        params: Dict[str, List],
        max_combos: int = 10
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations for an operator."""
        if not params:
            return [{}]
        
        combos = [{}]
        for param_name, values in params.items():
            new_combos = []
            for combo in combos:
                for val in values[:max_combos]:
                    new_combo = combo.copy()
                    new_combo[param_name] = val
                    new_combos.append(new_combo)
            combos = new_combos[:max_combos]
        
        return combos
    
    def is_valid_expansion(
        self,
        parent_op: Optional[Operator],
        child_action: Tuple[str, Any],
        arg_idx: int = 0
    ) -> bool:
        """
        Check if child action is valid given parent operator and argument position.
        
        Args:
            parent_op: Parent operator (None if at root)
            child_action: (action_type, value) tuple
            arg_idx: Which argument position this fills
            
        Returns:
            True if expansion is valid
        """
        if parent_op is None:
            return True  # Root can be anything
        
        action_type, value = child_action
        
        if action_type == 'operand':
            return parent_op.accepts_input(arg_idx, value.dim_type)
        elif action_type == 'operator':
            op, _ = value
            # Check if operator output is compatible with parent input
            output_dim = op.get_output_dim([])  # Get default output dim
            return parent_op.accepts_input(arg_idx, output_dim)
        
        return False
    
    def expand_node(
        self,
        action: Tuple[str, Any],
        children: List[ASTNode] = None
    ) -> ASTNode:
        """
        Create an AST node from an action.
        
        Args:
            action: (action_type, value) tuple
            children: Child nodes for operator actions
            
        Returns:
            New ASTNode
        """
        action_type, value = action
        
        if action_type == 'operand':
            return ASTNode(
                node_type='operand',
                value=value,
                dim_type=value.dim_type
            )
        elif action_type == 'operator':
            op, params = value
            node_children = list(children) if children else []
            
            # Add parameter nodes
            for param_name, param_val in params.items():
                node_children.append(ASTNode(
                    node_type='param',
                    value=f"{param_name}={param_val}"
                ))
            
            return ASTNode(
                node_type='operator',
                value=op,
                children=node_children
            )
        elif action_type == 'constant':
            return ASTNode(
                node_type='constant',
                value=value,
                dim_type=DimType.RATIO
            )
        
        raise ValueError(f"Unknown action type: {action_type}")
    
    def compute_ast_similarity(self, ast1: ASTNode, ast2: ASTNode) -> float:
        """
        Compute structural similarity between two ASTs.
        
        Uses tree edit distance normalized to [0, 1].
        Higher value means more similar.
        """
        # Simple approach: compare structure hashes at each level
        def get_structure(node: ASTNode) -> str:
            if node.node_type == 'operand':
                return f"O:{node.value.name}"
            elif node.node_type == 'constant':
                return f"C"
            elif node.node_type == 'operator':
                children_str = ",".join(get_structure(c) for c in node.children 
                                       if c.node_type != 'param')
                return f"{node.value.name}({children_str})"
            return ""
        
        s1 = get_structure(ast1)
        s2 = get_structure(ast2)
        
        # Simple Jaccard similarity on character n-grams
        def ngrams(s, n=3):
            return set(s[i:i+n] for i in range(len(s) - n + 1))
        
        ng1 = ngrams(s1)
        ng2 = ngrams(s2)
        
        if not ng1 or not ng2:
            return 0.0
        
        intersection = len(ng1 & ng2)
        union = len(ng1 | ng2)
        
        return intersection / union if union > 0 else 0.0


def build_expression_from_ast(node: ASTNode) -> str:
    """
    Convert an AST to a Brain expression string.
    
    Args:
        node: Root AST node
        
    Returns:
        Valid Brain expression string
    """
    if node.node_type == 'operand':
        return node.value.name
    
    elif node.node_type == 'constant':
        return str(node.value)
    
    elif node.node_type == 'param':
        return str(node.value)
    
    elif node.node_type == 'operator':
        op = node.value
        
        # Separate children from params (skip None placeholders)
        child_exprs = []
        param_strs = []
        
        for child in node.children:
            if child is None:
                # Incomplete expression - return None or placeholder
                return None
            if child.node_type == 'param':
                param_strs.append(str(child.value))
            else:
                child_expr = build_expression_from_ast(child)
                if child_expr is None:
                    return None  # Propagate incomplete status
                child_exprs.append(child_expr)
        
        # Build expression using operator syntax
        syntax = op.syntax
        
        # Replace positional arguments
        for i, expr in enumerate(child_exprs):
            syntax = syntax.replace(f"{{{i}}}", expr)
        
        # Replace named parameters
        for param_str in param_strs:
            if '=' in param_str:
                name, val = param_str.split('=', 1)
                syntax = syntax.replace(f"{{{name}}}", val)
        
        return syntax
    
    return ""


def create_default_operands(field_list: List[str], dataset: str = "model110") -> List[Operand]:
    """
    Create Operand objects from a list of field names.
    
    Args:
        field_list: List of field names
        dataset: Dataset name
        
    Returns:
        List of Operand objects with inferred dimension types
    """
    operands = []
    
    # Dimension inference based on field name patterns
    for field in field_list:
        field_lower = field.lower()
        
        if any(x in field_lower for x in ['price', 'close', 'open', 'high', 'low']):
            dim = DimType.PRICE
        elif any(x in field_lower for x in ['volume', 'vol', 'turnover']):
            dim = DimType.VOLUME
        elif any(x in field_lower for x in ['return', 'growth', 'change', 'pct', 'ratio']):
            dim = DimType.RATIO
        elif any(x in field_lower for x in ['rank', 'percentile']):
            dim = DimType.RANK
        elif any(x in field_lower for x in ['count', 'num']):
            dim = DimType.COUNT
        else:
            dim = DimType.RATIO  # Default to ratio for fundamental fields
        
        operands.append(Operand(
            name=field,
            dim_type=dim,
            dataset=dataset
        ))
    
    return operands
