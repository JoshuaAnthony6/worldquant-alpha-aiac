#!/usr/bin/env python3
"""
RL Training Script for Alpha Mining

Implements PPO training loop with:
- Environment initialization with Brain API
- Actor-Critic network training
- Checkpointing and early stopping
- TensorBoard logging
- Beam/MCTS search integration

Usage:
    python train.py --episodes 1000 --beam-width 50 --log-dir logs

Reference: Plan Section 6 - Training Pipeline
"""

import os
import sys
import argparse
import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
try:
    import ace_lib as ace
except ImportError:
    logger.warning("ace_lib not available, running in mock mode")
    ace = None

from optimization_chain import _generate_window_variants as generate_window_sweep, generate_settings_variants as generate_settings_sweep

from rl_alpha.grammar import (
    ExpressionGrammar, 
    OPERATORS,
    create_default_operands,
    build_expression_from_ast
)
from rl_alpha.env import AlphaEnv, EnvConfig, create_env
from rl_alpha.search import BeamSearchEngine, MCTSEngine, create_search_engine
from rl_alpha.logger import AlphaLogger, create_logger
from alpha_scoring import (
    calculate_alpha_score,
    calculate_consultant_score,
    AlphaPool,
    calculate_rl_reward,
    is_submission_ready,
    SHARPE_THRESHOLD,
    FITNESS_THRESHOLD,
    MARGIN_THRESHOLD
)

# Optional torch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available, RL training disabled")


class PPOTrainer:
    """
    PPO trainer for alpha generation policy.
    
    Implements Proximal Policy Optimization with:
    - Clipped objective
    - Value function loss
    - Entropy bonus
    - GAE advantage estimation
    """
    
    def __init__(
        self,
        env: AlphaEnv,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize PPO trainer.
        
        Args:
            env: Alpha generation environment
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clip epsilon
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm
            device: Device to use (cpu/cuda)
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for PPO training")
        
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        
        # Create actor-critic network
        from rl_alpha.policy import create_actor_critic
        self.model = create_actor_critic(
            env,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training stats
        self.total_steps = 0
        self.episode_count = 0
    
    def collect_rollout(
        self,
        num_steps: int = 128
    ) -> Dict[str, torch.Tensor]:
        """
        Collect rollout data from environment.
        
        Args:
            num_steps: Number of steps to collect
            
        Returns:
            Dictionary of rollout tensors
        """
        obs_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        values_list = []
        dones_list = []
        
        obs, info = self.env.reset()
        
        for _ in range(num_steps):
            # Convert observation to tensors
            obs_tensor = self._obs_to_tensor(obs)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, entropy, value = self.model(obs_tensor)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action.item())
            done = terminated or truncated
            
            # Store transition
            obs_list.append(obs_tensor)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            rewards_list.append(torch.tensor([reward], dtype=torch.float32))
            values_list.append(value)
            dones_list.append(torch.tensor([float(done)]))
            
            if done:
                obs, info = self.env.reset()
                self.episode_count += 1
            else:
                obs = next_obs
            
            self.total_steps += 1
        
        # Compute advantages with GAE
        returns, advantages = self._compute_gae(
            rewards_list, values_list, dones_list
        )
        
        return {
            'obs': obs_list,
            'actions': torch.cat(actions_list),
            'log_probs': torch.cat(log_probs_list),
            'returns': returns,
            'advantages': advantages,
            'values': torch.cat(values_list)
        }
    
    def _compute_gae(
        self,
        rewards: List[torch.Tensor],
        values: List[torch.Tensor],
        dones: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        rewards = torch.cat(rewards)
        values = torch.cat(values)
        dones = torch.cat(dones)
        
        T = len(rewards)
        advantages = torch.zeros(T)
        last_gae = 0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0  # Terminal
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(
        self,
        rollout: Dict[str, torch.Tensor],
        num_epochs: int = 4,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Perform PPO update.
        
        Args:
            rollout: Rollout data
            num_epochs: Number of update epochs
            batch_size: Mini-batch size
            
        Returns:
            Training statistics
        """
        obs_list = rollout['obs']
        actions = rollout['actions']
        old_log_probs = rollout['log_probs']
        returns = rollout['returns']
        advantages = rollout['advantages']
        
        T = len(actions)
        indices = np.arange(T)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        num_updates = 0
        
        for _ in range(num_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_obs = [obs_list[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluate actions
                batch_obs_combined = self._combine_obs(batch_obs)
                log_probs, entropy, values = self.model.evaluate(
                    batch_obs_combined, batch_actions
                )
                
                # Policy loss (clipped)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track stats
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                
                # Approximate KL divergence
                with torch.no_grad():
                    kl = (batch_old_log_probs - log_probs).mean().item()
                    total_kl += kl
                
                num_updates += 1
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'kl_divergence': total_kl / num_updates
        }
    
    def _obs_to_tensor(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """Convert observation dict to tensors."""
        return {
            'ast': torch.tensor(obs['ast'], dtype=torch.float32, device=self.device),
            'cursor': obs['cursor'],
            'depth': obs['depth'],
            'valid_action_mask': torch.tensor(
                obs['valid_action_mask'], dtype=torch.float32, device=self.device
            )
        }
    
    def _combine_obs(self, obs_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """Combine list of observations into batched tensors."""
        return {
            'ast': torch.stack([o['ast'] for o in obs_list]),
            'cursor': torch.tensor([o['cursor'] if isinstance(o['cursor'], int) 
                                   else o['cursor'].item() for o in obs_list]),
            'depth': torch.tensor([o['depth'] if isinstance(o['depth'], int)
                                  else o['depth'].item() for o in obs_list]),
            'valid_action_mask': torch.stack([o['valid_action_mask'] for o in obs_list])
        }
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']
        logger.info(f"Loaded checkpoint from {path}")


def train_with_search(
    brain_session,
    field_list: List[str],
    dataset: str = "model110",
    region: str = "USA",
    universe: str = "TOP3000",
    num_iterations: int = 100,
    candidates_per_iter: int = 100,
    search_method: str = "beam",
    beam_width: int = 50,
    log_dir: str = "logs",
    checkpoint_dir: str = "checkpoints"
):
    """
    Train using search-based approach (no explicit RL).
    
    Uses beam/MCTS search with heuristic scoring.
    
    Args:
        brain_session: Brain API session
        field_list: Available data fields
        dataset: Dataset name
        region: Market region
        universe: Stock universe
        num_iterations: Number of search iterations
        candidates_per_iter: Candidates per iteration
        search_method: "beam" or "mcts"
        beam_width: Beam search width
        log_dir: Log directory
        checkpoint_dir: Checkpoint directory
    """
    logger.info(f"Starting search-based training with {search_method} search")
    
    # Initialize components
    operands = create_default_operands(field_list, dataset)
    grammar = ExpressionGrammar(operators=OPERATORS, operands=operands)
    alpha_pool = AlphaPool(max_size=1000)
    
    # Initialize logger
    alpha_logger = create_logger(
        log_dir=log_dir,
        experiment_name=f"{search_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Create search engine with FAST proxy evaluation (no Brain API during search)
    # Proxy scoring tuned for CONSULTANT CRITERIA:
    # - Simple expressions (≤6 ops, ≤3 fields) = STRONG bonus
    # - group_rank performs best empirically
    # - ts_mean(rank(ts_zscore(...))) = our best pattern
    
    # Fields with historically high performance (ordered by Sharpe potential)
    # quality had Sharpe=1.41, alternative had Sharpe=0.87
    HIGH_VALUE_FIELDS = ['quality', 'analyst_sentiment', 'growth', 'tree', 'alternative']
    BEST_FIELD = 'quality'  # mdl110_quality consistently high Sharpe
    
    def proxy_evaluate_fn(expression: str) -> float:
        """Fast heuristic score for guiding search (no API call)."""
        import re
        
        score = 0.3  # Base score
        expr_lower = expression.lower()
        
        # Count operators and fields for complexity check
        op_count = len(re.findall(r'[a-z_]+\s*\(', expr_lower))
        field_count = len(set(re.findall(r'mdl\d+_[a-z_]+', expr_lower)))
        
        # CRITICAL: Complexity scoring (consultant requires ≤6 ops, ≤3 fields)
        if op_count <= 3 and field_count <= 1:
            score += 0.4  # Very simple - big bonus
        elif op_count <= 6 and field_count <= 3:
            score += 0.2  # Meets consultant criteria
        else:
            score -= 0.3  # Too complex - heavy penalty
        
        # Pattern bonuses (based on observed results)
        # group_rank with sector/subindustry scored 1.0+
        if 'group_rank' in expr_lower:
            if 'sector' in expr_lower:
                score += 0.3
            elif 'subindustry' in expr_lower:
                score += 0.25
        
        # rank + ts_* combos scored well
        if 'rank' in expr_lower and any(f in expression for f in ['ts_zscore', 'ts_mean', 'ts_ir']):
            score += 0.2
        elif 'rank' in expr_lower:
            score += 0.1
        
        # ts_ir is powerful (information ratio)
        if 'ts_ir' in expr_lower:
            score += 0.15
        
        # High-value field bonus
        if BEST_FIELD in expr_lower:
            score += 0.2  # Extra bonus for quality (highest Sharpe)
        else:
            for field in HIGH_VALUE_FIELDS:
                if field in expr_lower:
                    score += 0.1
                    break
        
        return min(score, 1.0)
    
    def brain_evaluate_fn(expression: str) -> float:
        """Full evaluation using Brain API."""
        if brain_session is None:
            return proxy_evaluate_fn(expression)
        
        try:
            config = ace.generate_alpha(
                regular=expression,
                alpha_type="REGULAR",
                region=region,
                universe=universe,
                delay=1,
                neutralization="INDUSTRY",
                decay=4,
                truncation=0.02,
                pasteurization="ON",
                test_period="P2Y",
                visualization=False
            )
            
            results = ace.simulate_alpha_list_multi(
                brain_session, [config],
                limit_of_concurrent_simulations=4,
                limit_of_multi_simulations=4
            )
            
            if results and results[0].get('alpha_id'):
                return calculate_alpha_score(results[0])
            return -1.0
        except Exception as e:
            logger.debug(f"Evaluation failed: {e}")
            return -1.0
    
    # Use FAST proxy during search, real API evaluation happens in batch later
    search_engine = create_search_engine(
        grammar=grammar,
        method=search_method,
        beam_width=beam_width,
        evaluate_fn=proxy_evaluate_fn  # Fast heuristic, not Brain API
    )
    
    # Training loop
    best_score = float('-inf')
    best_expression = ""
    best_details = {}  # Best alpha's detailed metrics
    total_alphas = 0
    submittable_count = 0
    evaluated_expressions = set()  # Track all evaluated to avoid repeats
    
    try:
        for iteration in range(num_iterations):
            start_time = time.time()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            logger.info(f"{'='*60}")
            
            # Run search (fast, uses proxy evaluation)
            # Use reset_diversity=False to accumulate seen hashes across iterations
            candidates = search_engine.search(
                num_candidates=candidates_per_iter,
                reset_diversity=False,  # Keep seen_hashes to avoid repeats
                iteration_seed=iteration  # Use iteration for randomization
            )
            
            search_duration_ms = int((time.time() - start_time) * 1000)
            logger.info(f"  Search generated {len(candidates)} candidates in {search_duration_ms}ms")
            
            # Filter out already evaluated expressions
            candidates = [(a, e, s) for a, e, s in candidates if e not in evaluated_expressions]
            if len(candidates) == 0:
                logger.info("  No new candidates - all already evaluated")
                continue
            
            logger.info(f"  {len(candidates)} new candidates after filtering")
            
            # Batch evaluate top candidates using Brain API
            if brain_session and candidates:
                # Sort by proxy score and take top for Brain evaluation
                eval_batch_size = min(10, len(candidates))  # Limit to 10 per iteration
                top_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)[:eval_batch_size]
                
                # Log expressions being evaluated
                logger.info(f"  Top {len(top_candidates)} expressions for Brain eval:")
                for i, (ast, expr, score) in enumerate(top_candidates[:3]):
                    logger.info(f"    [{i}] (proxy={score:.3f}) {expr[:80]}...")
                
                # Build configs for batch simulation
                # NOTE: decay=2 (not 4) to keep turnover in 5-20% range
                # Higher decay = lower turnover, we were getting 4.3% with decay=4
                configs = []
                for ast, expression, proxy_score in top_candidates:
                    try:
                        config = ace.generate_alpha(
                            regular=expression,
                            alpha_type="REGULAR",
                            region=region,
                            universe=universe,
                            delay=1,
                            neutralization="INDUSTRY",
                            decay=2,  # Lower decay for higher turnover
                            truncation=0.02,
                            pasteurization="ON",
                            test_period="P2Y",
                            visualization=False
                        )
                        configs.append((ast, expression, config))
                    except Exception as e:
                        logger.warning(f"Config generation failed for '{expression[:50]}': {e}")
                
                # Batch simulate with increased parallelism
                if configs:
                    try:
                        results = ace.simulate_alpha_list_multi(
                            brain_session,
                            [c[2] for c in configs],
                            limit_of_concurrent_simulations=5,  # Increased from 3
                            limit_of_multi_simulations=min(15, len(configs))  # Increased from 10
                        )
                        
                        # Update candidates with CONSULTANT scoring
                        candidates = []
                        for i, result in enumerate(results):
                            ast, expression, _ = configs[i]
                            if result and result.get('alpha_id'):
                                # 使用顾问标准评分
                                consultant_score, details = calculate_consultant_score(result, expression)
                                candidates.append((ast, expression, consultant_score, result, details))
                                
                                # 详细日志: 显示关键指标
                                sharpe = details['sharpe']
                                fitness = details['fitness']
                                margin = details['margin']
                                turnover = details['turnover']
                                
                                status = "✓" if details['is_submittable'] else "✗"
                                logger.info(
                                    f"    {status} S={sharpe:.2f} F={fitness:.2f} M={margin*10000:.1f}bp "
                                    f"T={turnover*100:.1f}% | Score={consultant_score:.1f} | "
                                    f"{expression[:50]}..."
                                )
                                
                                # 高亮可提交的alpha
                                if details['is_submittable']:
                                    submittable_count += 1
                                    logger.info(f"    🎯 SUBMITTABLE: {expression}")
                                    
                    except Exception as e:
                        logger.warning(f"Batch simulation failed: {e}")
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Process candidates
            iter_scores = []
            for item in candidates:
                if len(item) == 5:
                    ast, expression, score, result, details = item
                else:
                    ast, expression, score = item
                    details = {}
                    result = {}
                
                total_alphas += 1
                iter_scores.append(score)
                
                # Mark as evaluated to avoid repeats
                evaluated_expressions.add(expression)
                
                # Log to alpha pool
                if score > 0:
                    structure_sig = ast.hash() if ast else ""
                    alpha_pool.add(expression, structure_sig=structure_sig, score=score)
                
                # Track best
                if score > best_score:
                    best_score = score
                    best_expression = expression
                    best_details = details
                    logger.info(f"  ⭐ New best: {score:.1f} - {expression[:60]}...")
                
                # Log alpha
                alpha_logger.log_alpha(
                    expression=expression,
                    sim_result={'raw_score': score},
                    ast_hash=ast.hash() if ast else "",
                    search_method=search_method,
                    generation_time_ms=duration_ms // max(1, len(candidates))
                )
            
            # Log episode
            alpha_logger.log_episode(
                expression=best_expression,
                reward=best_score,
                num_steps=len(candidates),
                is_complete=True,
                search_method=search_method,
                duration_ms=duration_ms
            )
            
            # Log training stats
            if iter_scores:
                alpha_logger.log_training_step(
                    step=iteration,
                    policy_loss=0,
                    value_loss=0,
                    entropy=0,
                    avg_reward=np.mean(iter_scores),
                    max_reward=np.max(iter_scores),
                    min_reward=np.min(iter_scores),
                    success_rate=sum(1 for s in iter_scores if s > 0) / len(iter_scores)
                )
            
            # Progress report
            logger.info(f"  Candidates: {len(candidates)}")
            logger.info(f"  Best this iter: {max(iter_scores) if iter_scores else 0:.4f}")
            logger.info(f"  Avg this iter: {np.mean(iter_scores) if iter_scores else 0:.4f}")
            logger.info(f"  Overall best: {best_score:.4f}")
            logger.info(f"  Pool size: {len(alpha_pool.alphas)}")
            
            # Pool stats
            pool_stats = alpha_pool.get_stats()
            alpha_logger.log_pool_stats(
                pool_size=pool_stats['size'],
                avg_score=pool_stats['avg_score'],
                max_score=pool_stats['max_score'],
                submittable_count=submittable_count
            )
            
            # ============================================================
            # Optimization Chain: Apply parameter sweeps to high-scoring alphas
            # Focus on SETTINGS that affect Sharpe/Turnover: decay, neutralization
            # ============================================================
            # Extract scores from candidates (handle both 3-tuple and 5-tuple)
            high_score_alphas = []
            for item in candidates:
                if len(item) == 5:
                    a, e, s, r, d = item
                    high_score_alphas.append((a, e, s, d))
                elif len(item) >= 3:
                    a, e, s = item[:3]
                    high_score_alphas.append((a, e, s, {}))
            
            # Filter to high-scoring ones (score > 50 on consultant scale)
            high_score_alphas = [x for x in high_score_alphas if x[2] > 50]
            
            if high_score_alphas and brain_session:
                logger.info(f"  Optimizing {len(high_score_alphas)} high-score alphas with settings sweep...")
                
                for ast, expression, base_score, details in high_score_alphas[:3]:  # Top 3 only
                    # Skip if already optimized
                    if f"OPT:{expression}" in evaluated_expressions:
                        continue
                    evaluated_expressions.add(f"OPT:{expression}")
                    
                    # CRITICAL SETTINGS for consultant criteria:
                    # - Low decay (0-2) = Higher turnover (we need 5-20%, currently 4.3%)
                    # - Different neutralization = Different Sharpe/Fitness
                    # - Lower truncation = Higher margin but more risk
                    
                    # Current alpha issue: Turnover too low (4.3%), need 5%+
                    # Solution: Try decay=0,1,2 instead of decay=4
                    
                    all_variants = []
                    
                    # Settings variants focused on improving weak metrics
                    settings_to_try = [
                        # Lower decay to increase turnover
                        {'neutralization': 'INDUSTRY', 'decay': 0, 'truncation': 0.02},
                        {'neutralization': 'INDUSTRY', 'decay': 1, 'truncation': 0.02},
                        {'neutralization': 'INDUSTRY', 'decay': 2, 'truncation': 0.02},
                        # Different neutralization for better Sharpe
                        {'neutralization': 'SECTOR', 'decay': 2, 'truncation': 0.02},
                        {'neutralization': 'SUBINDUSTRY', 'decay': 2, 'truncation': 0.02},
                        {'neutralization': 'MARKET', 'decay': 2, 'truncation': 0.02},
                        # Lower truncation for higher margin
                        {'neutralization': 'INDUSTRY', 'decay': 2, 'truncation': 0.01},
                        {'neutralization': 'INDUSTRY', 'decay': 2, 'truncation': 0.03},
                    ]
                    
                    for settings in settings_to_try:
                        all_variants.append({'expression': expression, 'settings': settings})
                    
                    # Also try window variants if expression has window params
                    window_variants = generate_window_sweep(expression)[:3]
                    for wv in window_variants:
                        all_variants.append({'expression': wv.get('expression', expression), 'settings': {'neutralization': 'INDUSTRY', 'decay': 2, 'truncation': 0.02}})
                    
                    if not all_variants:
                        continue
                    
                    # Build configs for variants
                    opt_configs = []
                    default_settings = {'neutralization': 'INDUSTRY', 'decay': 2, 'truncation': 0.02}
                    for variant in all_variants:
                        var_expr = variant.get('expression', expression)
                        var_settings = variant.get('settings', default_settings)
                        
                        if var_expr in evaluated_expressions:
                            continue
                        
                        try:
                            config = ace.generate_alpha(
                                regular=var_expr,
                                alpha_type="REGULAR",
                                region=region,
                                universe=universe,
                                delay=1,
                                neutralization=var_settings.get('neutralization', 'INDUSTRY'),
                                decay=var_settings.get('decay', 4),
                                truncation=var_settings.get('truncation', 0.02),
                                pasteurization="ON",
                                test_period="P2Y",
                                visualization=False
                            )
                            opt_configs.append((var_expr, config, var_settings))
                        except Exception:
                            pass
                    
                    # Batch simulate variants
                    if opt_configs:
                        try:
                            results = ace.simulate_alpha_list_multi(
                                brain_session,
                                [c[1] for c in opt_configs],
                                limit_of_concurrent_simulations=3,
                                limit_of_multi_simulations=min(10, len(opt_configs))
                            )
                            
                            for i, result in enumerate(results):
                                var_expr, _, var_settings = opt_configs[i]
                                evaluated_expressions.add(var_expr)
                                
                                if result and result.get('alpha_id'):
                                    opt_score = calculate_alpha_score(result)
                                    total_alphas += 1
                                    
                                    if opt_score > base_score:
                                        logger.info(f"    Optimization improved: {base_score:.4f} -> {opt_score:.4f}")
                                        logger.info(f"      {var_expr[:60]}...")
                                        
                                        if opt_score > best_score:
                                            best_score = opt_score
                                            best_expression = var_expr
                                            logger.info(f"  *** New overall best: {opt_score:.4f}")
                                        
                                        alpha_pool.add(var_expr, score=opt_score)
                        except Exception as e:
                            logger.debug(f"Optimization batch failed: {e}")
            
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    
    finally:
        # Final report
        alpha_logger.generate_daily_report()
        alpha_logger.close()
        
        logger.info(f"\n{'='*60}")
        logger.info("Training Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total alphas generated: {total_alphas}")
        logger.info(f"Submittable alphas: {submittable_count}")
        logger.info(f"Best score: {best_score:.1f}")
        logger.info(f"Best expression: {best_expression}")
        
        # 顾问标准报告
        if best_details:
            logger.info(f"\n--- CONSULTANT CRITERIA CHECK ---")
            logger.info(f"Sharpe:   {best_details.get('sharpe', 0):.3f}  (need >1.58) {'✓' if best_details.get('meets_sharpe') else '✗'}")
            logger.info(f"Fitness:  {best_details.get('fitness', 0):.3f}  (need >1.0)  {'✓' if best_details.get('meets_fitness') else '✗'}")
            logger.info(f"Margin:   {best_details.get('margin', 0)*10000:.1f} bp (need >15bp) {'✓' if best_details.get('meets_margin') else '✗'}")
            logger.info(f"Turnover: {best_details.get('turnover', 0)*100:.1f}% (need 5-20%) {'✓' if best_details.get('meets_turnover') else '✗'}")
            logger.info(f"Returns:  {best_details.get('returns', 0)*100:.1f}%  (need >5%)  {'✓' if best_details.get('meets_returns') else '✗'}")
            logger.info(f"Ops:      {best_details.get('op_count', 0)}     (need ≤6)")
            logger.info(f"Fields:   {best_details.get('field_count', 0)}     (need ≤3)")
            
            if best_details.get('is_submittable'):
                logger.info(f"\n🎯 ALPHA IS SUBMITTABLE!")


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="RL Training for Alpha Mining")
    
    # Training parameters
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of training iterations")
    parser.add_argument("--candidates", type=int, default=100,
                       help="Candidates per iteration")
    parser.add_argument("--method", type=str, default="beam",
                       choices=["beam", "mcts", "hybrid"],
                       help="Search method")
    parser.add_argument("--beam-width", type=int, default=50,
                       help="Beam search width")
    
    # Environment parameters
    parser.add_argument("--dataset", type=str, default="model110",
                       help="Dataset ID")
    parser.add_argument("--region", type=str, default="USA",
                       help="Market region")
    parser.add_argument("--universe", type=str, default="TOP3000",
                       help="Stock universe")
    
    # Logging parameters
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Checkpoint directory")
    
    # Mode
    parser.add_argument("--mock", action="store_true",
                       help="Run in mock mode without Brain API")
    
    args = parser.parse_args()
    
    # Initialize Brain session
    brain_session = None
    field_list = []
    
    if not args.mock and ace is not None:
        try:
            brain_session = ace.start_session()
            if ace.check_session_timeout(brain_session) < 3000:
                brain_session = ace.check_session_and_relogin(brain_session)
            
            # Get field list
            fields_df = ace.get_datafields(
                brain_session,
                dataset_id=args.dataset,
                region=args.region,
                universe=args.universe
            )
            field_list = fields_df['id'].tolist()
            logger.info(f"Loaded {len(field_list)} fields from {args.dataset}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Brain session: {e}")
            logger.info("Falling back to mock mode")
            args.mock = True
    
    if args.mock or not field_list:
        # Use mock field list
        field_list = [
            f"mdl110_{name}" for name in [
                "growth", "value", "quality", "momentum", "volatility",
                "size", "leverage", "profitability", "investment", "trading"
            ]
        ]
        logger.info(f"Using mock field list with {len(field_list)} fields")
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Run training
    train_with_search(
        brain_session=brain_session,
        field_list=field_list,
        dataset=args.dataset,
        region=args.region,
        universe=args.universe,
        num_iterations=args.iterations,
        candidates_per_iter=args.candidates,
        search_method=args.method,
        beam_width=args.beam_width,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir
    )


if __name__ == "__main__":
    main()
