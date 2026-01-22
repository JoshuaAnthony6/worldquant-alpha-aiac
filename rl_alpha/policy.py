"""
PPO Policy and Value Networks for Alpha Generation

Implements:
- Transformer encoder for AST representation
- MLP policy head with masked softmax
- Value head for advantage estimation

Reference: Alpha² + PPO (Proximal Policy Optimization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence data."""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ASTEncoder(nn.Module):
    """
    Transformer-based encoder for AST representation.
    
    Takes the flattened AST encoding and produces a fixed-size
    representation suitable for policy/value heads.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_nodes: int = 12
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_nodes, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Attention pooling to get fixed-size output
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        ast_encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode AST into fixed-size representation.
        
        Args:
            ast_encoding: (batch, max_nodes, input_dim) AST encoding
            mask: (batch, max_nodes) optional padding mask
            
        Returns:
            (batch, hidden_dim) encoded representation
        """
        # Project to hidden dim
        x = self.input_proj(ast_encoding)  # (batch, nodes, hidden)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=~mask.bool())
        else:
            x = self.transformer(x)
        
        # Attention pooling
        attn_weights = self.attention_pool(x)  # (batch, nodes, 1)
        pooled = torch.sum(x * attn_weights, dim=1)  # (batch, hidden)
        
        return pooled


class AlphaPolicyNetwork(nn.Module):
    """
    Policy network for alpha generation.
    
    Takes AST encoding and outputs action probabilities
    with masking for invalid actions.
    """
    
    def __init__(
        self,
        ast_input_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_nodes: int = 12
    ):
        super().__init__()
        
        self.num_actions = num_actions
        
        # AST encoder
        self.ast_encoder = ASTEncoder(
            input_dim=ast_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_nodes=max_nodes
        )
        
        # Additional context features (cursor position, depth)
        self.context_embed = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to get action and log probability.
        
        Args:
            obs: Observation dict with 'ast', 'cursor', 'depth', 'valid_action_mask'
            deterministic: If True, return argmax action
            
        Returns:
            (action, log_prob, entropy)
        """
        ast = obs['ast']  # (batch, nodes, features)
        cursor = obs['cursor']  # (batch,) or scalar
        depth = obs['depth']  # (batch,) or scalar
        valid_mask = obs['valid_action_mask']  # (batch, num_actions)
        
        # Ensure batch dimension
        if ast.dim() == 2:
            ast = ast.unsqueeze(0)
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)
        
        batch_size = ast.size(0)
        
        # Encode AST
        ast_features = self.ast_encoder(ast)  # (batch, hidden)
        
        # Encode context
        if isinstance(cursor, int):
            cursor = torch.tensor([cursor], device=ast.device)
        if isinstance(depth, int):
            depth = torch.tensor([depth], device=ast.device)
        
        cursor = cursor.float().view(-1, 1) / 12.0  # Normalize
        depth = depth.float().view(-1, 1) / 6.0
        context = torch.cat([cursor, depth], dim=1)
        context_features = self.context_embed(context)  # (batch, hidden//4)
        
        # Combine features
        features = torch.cat([ast_features, context_features], dim=1)
        
        # Get logits
        logits = self.policy_head(features)  # (batch, num_actions)
        
        # Apply action mask (set invalid actions to -inf)
        valid_mask = valid_mask.float()
        masked_logits = logits + (1 - valid_mask) * (-1e9)
        
        # Get action distribution
        probs = F.softmax(masked_logits, dim=-1)
        dist = Categorical(probs)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.
        
        Args:
            obs: Observation dict
            actions: Actions to evaluate
            
        Returns:
            (log_probs, entropy)
        """
        ast = obs['ast']
        cursor = obs['cursor']
        depth = obs['depth']
        valid_mask = obs['valid_action_mask']
        
        # Ensure batch dimension
        if ast.dim() == 2:
            ast = ast.unsqueeze(0)
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)
        
        # Encode AST
        ast_features = self.ast_encoder(ast)
        
        # Encode context
        if isinstance(cursor, int):
            cursor = torch.tensor([cursor], device=ast.device)
        if isinstance(depth, int):
            depth = torch.tensor([depth], device=ast.device)
        
        cursor = cursor.float().view(-1, 1) / 12.0
        depth = depth.float().view(-1, 1) / 6.0
        context = torch.cat([cursor, depth], dim=1)
        context_features = self.context_embed(context)
        
        features = torch.cat([ast_features, context_features], dim=1)
        logits = self.policy_head(features)
        
        # Apply mask
        valid_mask = valid_mask.float()
        masked_logits = logits + (1 - valid_mask) * (-1e9)
        
        probs = F.softmax(masked_logits, dim=-1)
        dist = Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy


class AlphaValueNetwork(nn.Module):
    """
    Value network for advantage estimation.
    
    Estimates state value for PPO critic.
    """
    
    def __init__(
        self,
        ast_input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_nodes: int = 12
    ):
        super().__init__()
        
        # AST encoder (can share with policy or be separate)
        self.ast_encoder = ASTEncoder(
            input_dim=ast_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_nodes=max_nodes
        )
        
        # Context embedding
        self.context_embed = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to get state value.
        
        Args:
            obs: Observation dict
            
        Returns:
            (batch,) value estimates
        """
        ast = obs['ast']
        cursor = obs['cursor']
        depth = obs['depth']
        
        # Ensure batch dimension
        if ast.dim() == 2:
            ast = ast.unsqueeze(0)
        
        # Encode AST
        ast_features = self.ast_encoder(ast)
        
        # Encode context
        if isinstance(cursor, int):
            cursor = torch.tensor([cursor], device=ast.device)
        if isinstance(depth, int):
            depth = torch.tensor([depth], device=ast.device)
        
        cursor = cursor.float().view(-1, 1) / 12.0
        depth = depth.float().view(-1, 1) / 6.0
        context = torch.cat([cursor, depth], dim=1)
        context_features = self.context_embed(context)
        
        features = torch.cat([ast_features, context_features], dim=1)
        value = self.value_head(features).squeeze(-1)
        
        return value


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    
    Shares the AST encoder between policy and value heads
    for more efficient learning.
    """
    
    def __init__(
        self,
        ast_input_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_nodes: int = 12
    ):
        super().__init__()
        
        self.num_actions = num_actions
        
        # Shared AST encoder
        self.ast_encoder = ASTEncoder(
            input_dim=ast_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_nodes=max_nodes
        )
        
        # Context embedding (shared)
        self.context_embed = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Smaller initialization for output layers
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
    
    def _encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observation to features."""
        ast = obs['ast']
        cursor = obs['cursor']
        depth = obs['depth']
        
        # Ensure batch dimension
        if ast.dim() == 2:
            ast = ast.unsqueeze(0)
        
        batch_size = ast.size(0)
        device = ast.device
        
        # Encode AST
        ast_features = self.ast_encoder(ast)
        
        # Encode context
        if isinstance(cursor, (int, np.integer)):
            cursor = torch.tensor([cursor], device=device, dtype=torch.float32)
        elif isinstance(cursor, torch.Tensor):
            cursor = cursor.float()
        else:
            cursor = torch.tensor(cursor, device=device, dtype=torch.float32)
        
        if isinstance(depth, (int, np.integer)):
            depth = torch.tensor([depth], device=device, dtype=torch.float32)
        elif isinstance(depth, torch.Tensor):
            depth = depth.float()
        else:
            depth = torch.tensor(depth, device=device, dtype=torch.float32)
        
        cursor = cursor.view(-1, 1) / 12.0
        depth = depth.view(-1, 1) / 6.0
        context = torch.cat([cursor, depth], dim=1)
        context_features = self.context_embed(context)
        
        return torch.cat([ast_features, context_features], dim=1)
    
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for both action selection and value estimation.
        
        Args:
            obs: Observation dict
            deterministic: If True, return argmax action
            
        Returns:
            (action, log_prob, entropy, value)
        """
        valid_mask = obs['valid_action_mask']
        
        # Ensure batch dimension for mask
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)
        
        # Get shared features
        features = self._encode(obs)
        
        # Policy
        logits = self.policy_head(features)
        valid_mask = valid_mask.float()
        masked_logits = logits + (1 - valid_mask) * (-1e9)
        probs = F.softmax(masked_logits, dim=-1)
        dist = Categorical(probs)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # Value
        value = self.value_head(features).squeeze(-1)
        
        return action, log_prob, entropy, value
    
    def evaluate(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            obs: Observation dict (batched)
            actions: Actions taken
            
        Returns:
            (log_probs, entropy, values)
        """
        valid_mask = obs['valid_action_mask']
        
        features = self._encode(obs)
        
        # Policy evaluation
        logits = self.policy_head(features)
        valid_mask = valid_mask.float()
        masked_logits = logits + (1 - valid_mask) * (-1e9)
        probs = F.softmax(masked_logits, dim=-1)
        dist = Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Value
        values = self.value_head(features).squeeze(-1)
        
        return log_probs, entropy, values
    
    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get value estimate only."""
        features = self._encode(obs)
        return self.value_head(features).squeeze(-1)


def create_actor_critic(
    env,
    hidden_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1
) -> ActorCritic:
    """
    Factory function to create ActorCritic network from environment.
    
    Args:
        env: AlphaEnv instance
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        
    Returns:
        Configured ActorCritic network
    """
    ast_space = env.observation_space['ast']
    ast_input_dim = ast_space.shape[1]
    max_nodes = ast_space.shape[0]
    num_actions = env.action_space.n
    
    return ActorCritic(
        ast_input_dim=ast_input_dim,
        num_actions=num_actions,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        max_nodes=max_nodes
    )
