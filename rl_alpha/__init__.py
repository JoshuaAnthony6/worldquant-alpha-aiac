"""
RL Alpha Mining Module

Reinforcement learning based alpha expression generation using:
- Alpha² style dimension-aware grammar
- Chain-of-Alpha optimization chains
- AlphaGen multi-objective rewards

References:
- Alpha²: Discovering Logical Formulaic Alphas using Deep RL (2024)
- Chain-of-Alpha: Factor Generation and Optimization Chains (2025)
- AlphaGen: RL for Quant Alpha Generation (KDD 2023)
"""

from .grammar import (
    DimType,
    Operator,
    Operand,
    ASTNode,
    ExpressionGrammar,
    build_expression_from_ast,
)
from .env import AlphaEnv
from .policy import AlphaPolicyNetwork, AlphaValueNetwork
from .search import MCTSNode, BeamSearchEngine
from .logger import AlphaLogger

__version__ = "1.0.0"
__all__ = [
    "DimType",
    "Operator", 
    "Operand",
    "ASTNode",
    "ExpressionGrammar",
    "build_expression_from_ast",
    "AlphaEnv",
    "AlphaPolicyNetwork",
    "AlphaValueNetwork",
    "MCTSNode",
    "BeamSearchEngine",
    "AlphaLogger",
]
