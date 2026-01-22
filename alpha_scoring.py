"""
Alpha Scoring Module

Implements multi-objective scoring for alpha evaluation based on Brain simulation results.
Replaces single-metric (Sharpe) evaluation with a composite score that reflects
Brain's actual pass/fail criteria.

Enhanced with:
- Pool correlation penalty (AlphaGen style)
- Diversity bonus for exploration
- Risk-adjusted metrics
- Submission readiness scoring

Reference: 优化.md Section 3.1, AlphaGen (KDD 2023), Alpha² (2024)
"""

from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass, field
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Constants and Thresholds
# ============================================================================

# 顾问提交标准 (Consultant Submission Thresholds)
SHARPE_THRESHOLD = 1.58      # IS Sharpe > 1.58
FITNESS_THRESHOLD = 1.0      # IS Fitness > 1.0
MARGIN_THRESHOLD = 0.0015    # Margin > 0.0015 (15 bps)
MARGIN_MIN = 0.0010          # Absolute minimum 10 bps
TURNOVER_MAX = 0.20          # Max 20%
TURNOVER_MIN = 0.05          # Min 5%
RETURNS_MIN = 0.05           # Min 5% returns
CORRELATION_THRESHOLD = 0.7

# IS-Ladder thresholds (D1)
LADDER_2Y_THRESHOLD = 1.8    # 2Y Sharpe > 1.8

# Scoring weights (optimized based on 优化.md recommendations)
DEFAULT_WEIGHTS = {
    'test_sharpe': 0.55,
    'train_sharpe': 0.25,
    'fitness': 0.20,
    'prod_corr_penalty': 0.30,
    'pool_corr_penalty': 0.20,  # NEW: correlation with existing pool
    'turnover_penalty': 0.15,
    'investability_penalty': 0.20,
    'diversity_bonus': 0.15,  # NEW: diversity bonus
    'test_pass_bonus': 0.10,  # NEW: bonus for passing tests
}


@dataclass
class AlphaMetrics:
    """Structured metrics for an alpha."""
    alpha_id: str = ""
    expression: str = ""
    
    # Performance metrics (IS = In-Sample/Train)
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_fitness: float = 0.0
    test_fitness: float = 0.0
    
    # 顾问关键指标 (Consultant Key Metrics)
    margin: float = 0.0           # Must be > 0.0015
    returns: float = 0.0          # Must be > 5%
    
    # Risk metrics
    turnover: float = 0.0
    drawdown: float = 0.0
    
    # Adjusted metrics
    risk_neutral_sharpe: float = 0.0
    invest_constrained_sharpe: float = 0.0
    
    # Correlation metrics
    prod_correlation: float = 0.0
    pool_correlation: float = 0.0
    self_correlation: float = 0.0
    
    # Test results
    tests_passed: int = 0
    tests_total: int = 0
    failed_tests: List[str] = field(default_factory=list)
    
    # Composite scores
    raw_score: float = 0.0
    adjusted_score: float = 0.0
    submission_ready: bool = False
    
    # 顾问标准检查 (Consultant criteria check)
    meets_sharpe: bool = False
    meets_fitness: bool = False
    meets_margin: bool = False
    meets_turnover: bool = False
    meets_returns: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'alpha_id': self.alpha_id,
            'expression': self.expression,
            'train_sharpe': self.train_sharpe,
            'test_sharpe': self.test_sharpe,
            'train_fitness': self.train_fitness,
            'test_fitness': self.test_fitness,
            'turnover': self.turnover,
            'risk_neutral_sharpe': self.risk_neutral_sharpe,
            'invest_constrained_sharpe': self.invest_constrained_sharpe,
            'prod_correlation': self.prod_correlation,
            'pool_correlation': self.pool_correlation,
            'tests_passed': self.tests_passed,
            'tests_total': self.tests_total,
            'failed_tests': self.failed_tests,
            'raw_score': self.raw_score,
            'adjusted_score': self.adjusted_score,
            'submission_ready': self.submission_ready
        }


class AlphaPool:
    """
    Manages a pool of alphas for diversity checking.
    
    Tracks:
    - Expression hashes for exact deduplication
    - Structural signatures for similarity checking
    - Correlation matrix approximation
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.alphas: List[Dict] = []
        self.expression_hashes: set = set()
        self.structure_signatures: set = set()
        self.recent_scores: List[float] = []
    
    def add(
        self,
        expression: str,
        ast_hash: str = None,
        structure_sig: str = None,
        score: float = 0.0,
        metrics: AlphaMetrics = None
    ) -> bool:
        """
        Add alpha to pool if unique.
        
        Returns:
            True if added, False if duplicate
        """
        # Check for exact duplicate
        expr_hash = hashlib.md5(expression.encode()).hexdigest()
        if expr_hash in self.expression_hashes:
            return False
        
        self.expression_hashes.add(expr_hash)
        
        if structure_sig:
            self.structure_signatures.add(structure_sig)
        
        self.alphas.append({
            'expression': expression,
            'expr_hash': expr_hash,
            'ast_hash': ast_hash,
            'structure_sig': structure_sig,
            'score': score,
            'metrics': metrics.to_dict() if metrics else {}
        })
        
        self.recent_scores.append(score)
        if len(self.recent_scores) > 100:
            self.recent_scores.pop(0)
        
        # Evict oldest if over capacity
        if len(self.alphas) > self.max_size:
            oldest = self.alphas.pop(0)
            self.expression_hashes.discard(oldest['expr_hash'])
        
        return True
    
    def compute_pool_correlation(
        self,
        expression: str,
        structure_sig: str = None
    ) -> float:
        """
        Estimate correlation with existing pool.
        
        Uses structural similarity as proxy for correlation
        when actual correlations aren't available.
        """
        if not self.alphas:
            return 0.0
        
        # Exact match check
        expr_hash = hashlib.md5(expression.encode()).hexdigest()
        if expr_hash in self.expression_hashes:
            return 1.0
        
        # Structural similarity
        if structure_sig and self.structure_signatures:
            max_sim = 0.0
            for existing_sig in self.structure_signatures:
                sim = self._jaccard_similarity(structure_sig, existing_sig)
                max_sim = max(max_sim, sim)
            return max_sim
        
        return 0.0
    
    def _jaccard_similarity(self, s1: str, s2: str, n: int = 3) -> float:
        """Compute Jaccard similarity on character n-grams."""
        if not s1 or not s2:
            return 0.0
        
        def ngrams(s):
            return set(s[i:i+n] for i in range(max(0, len(s) - n + 1)))
        
        ng1 = ngrams(s1)
        ng2 = ngrams(s2)
        
        if not ng1 or not ng2:
            return 0.0
        
        intersection = len(ng1 & ng2)
        union = len(ng1 | ng2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_diversity_bonus(
        self,
        expression: str,
        structure_sig: str = None
    ) -> float:
        """
        Calculate diversity bonus for a new alpha.
        
        Higher bonus for alphas that are more different from the pool.
        """
        pool_corr = self.compute_pool_correlation(expression, structure_sig)
        
        # Diversity bonus = 1 - correlation
        # Scale to [0, 1] range
        return max(0.0, 1.0 - pool_corr)
    
    def get_stats(self) -> Dict:
        """Get pool statistics."""
        return {
            'size': len(self.alphas),
            'unique_structures': len(self.structure_signatures),
            'avg_score': sum(self.recent_scores) / len(self.recent_scores) if self.recent_scores else 0.0,
            'max_score': max(self.recent_scores) if self.recent_scores else 0.0
        }


# ============================================================================
# Core Scoring Functions
# ============================================================================

def calculate_consultant_score(
    sim_result: Dict[str, Any],
    expression: str = ""
) -> Tuple[float, Dict[str, Any]]:
    """
    计算基于顾问提交标准的分数。
    
    顾问标准：
    - Sharpe > 1.58
    - Fitness > 1.0
    - Margin > 0.0015 (15 bps)
    - Turnover 5%-20%
    - Returns > 5% & Returns > drawdown
    - ≤6 operators, ≤3 fields
    
    Returns:
        (score, details) - score is 0-100, details contains all metrics
    """
    is_stats = _extract_is_stats(sim_result)
    os_stats = _extract_os_stats(sim_result)
    
    # 提取关键指标
    sharpe = _safe_float(is_stats.get('sharpe', is_stats.get('Sharpe', 0.0)))
    fitness = _safe_float(is_stats.get('fitness', is_stats.get('Fitness', 0.0)))
    turnover = _safe_float(is_stats.get('turnover', is_stats.get('Turnover', 0.0)))
    returns = _safe_float(is_stats.get('returns', is_stats.get('Returns', is_stats.get('return', 0.0))))
    drawdown = _safe_float(is_stats.get('drawdown', is_stats.get('Drawdown', is_stats.get('maxDrawdown', 0.0))))
    margin = _safe_float(is_stats.get('margin', is_stats.get('Margin', 0.0)))
    
    # OOS metrics
    os_sharpe = _safe_float(os_stats.get('sharpe', os_stats.get('Sharpe', 0.0)))
    
    # 表达式复杂度检查
    op_count = _count_operators(expression)
    field_count = _count_fields(expression)
    
    # 计算各项得分 (0-20 each, max 100)
    score = 0.0
    
    # 1. Sharpe score (0-25 points)
    if sharpe >= SHARPE_THRESHOLD:
        sharpe_score = 25.0
    elif sharpe >= 1.0:
        sharpe_score = 15.0 + 10.0 * (sharpe - 1.0) / (SHARPE_THRESHOLD - 1.0)
    elif sharpe > 0:
        sharpe_score = 15.0 * sharpe
    else:
        sharpe_score = 0.0
    score += sharpe_score
    
    # 2. Fitness score (0-20 points)
    if fitness >= FITNESS_THRESHOLD:
        fitness_score = 20.0
    elif fitness > 0:
        fitness_score = 20.0 * fitness
    else:
        fitness_score = 0.0
    score += fitness_score
    
    # 3. Margin score (0-20 points) - CRITICAL for consultant
    if margin >= MARGIN_THRESHOLD:
        margin_score = 20.0
    elif margin >= MARGIN_MIN:
        margin_score = 10.0 + 10.0 * (margin - MARGIN_MIN) / (MARGIN_THRESHOLD - MARGIN_MIN)
    elif margin > 0:
        margin_score = 10.0 * margin / MARGIN_MIN
    else:
        margin_score = 0.0
    score += margin_score
    
    # 4. Turnover score (0-15 points)
    if TURNOVER_MIN <= turnover <= TURNOVER_MAX:
        turnover_score = 15.0
    elif turnover < TURNOVER_MIN:
        turnover_score = max(0, 15.0 - 50.0 * (TURNOVER_MIN - turnover))
    else:  # turnover > TURNOVER_MAX
        turnover_score = max(0, 15.0 - 30.0 * (turnover - TURNOVER_MAX))
    score += turnover_score
    
    # 5. Returns vs Drawdown (0-10 points)
    if returns >= RETURNS_MIN and returns > abs(drawdown):
        returns_score = 10.0
    elif returns > 0 and returns > abs(drawdown):
        returns_score = 10.0 * returns / RETURNS_MIN
    else:
        returns_score = 0.0
    score += returns_score
    
    # 6. Complexity bonus (0-10 points) - simpler is better
    if op_count <= 6 and field_count <= 3:
        complexity_bonus = 10.0
    elif op_count <= 8 and field_count <= 4:
        complexity_bonus = 5.0
    else:
        complexity_bonus = 0.0
    score += complexity_bonus
    
    # 检查是否满足各项标准
    meets_sharpe = sharpe >= SHARPE_THRESHOLD
    meets_fitness = fitness >= FITNESS_THRESHOLD
    meets_margin = margin >= MARGIN_THRESHOLD
    meets_turnover = TURNOVER_MIN <= turnover <= TURNOVER_MAX
    meets_returns = returns >= RETURNS_MIN and returns > abs(drawdown)
    is_submittable = meets_sharpe and meets_fitness and meets_margin and meets_turnover
    
    details = {
        'sharpe': sharpe,
        'fitness': fitness,
        'margin': margin,
        'turnover': turnover,
        'returns': returns,
        'drawdown': drawdown,
        'os_sharpe': os_sharpe,
        'op_count': op_count,
        'field_count': field_count,
        'meets_sharpe': meets_sharpe,
        'meets_fitness': meets_fitness,
        'meets_margin': meets_margin,
        'meets_turnover': meets_turnover,
        'meets_returns': meets_returns,
        'is_submittable': is_submittable,
        'sharpe_score': sharpe_score,
        'fitness_score': fitness_score,
        'margin_score': margin_score,
        'turnover_score': turnover_score,
        'returns_score': returns_score,
        'complexity_bonus': complexity_bonus,
    }
    
    return score, details


def _count_operators(expression: str) -> int:
    """统计表达式中的运算符数量。"""
    import re
    # 匹配函数调用（运算符）
    ops = re.findall(r'[a-z_]+\s*\(', expression.lower())
    return len(ops)


def _count_fields(expression: str) -> int:
    """统计表达式中的数据字段数量。"""
    import re
    # 匹配 mdl110_xxx 或其他数据字段
    fields = set(re.findall(r'mdl\d+_[a-z_]+', expression.lower()))
    # 也匹配其他可能的字段
    other_fields = set(re.findall(r'\b(?:close|open|high|low|volume|returns|vwap)\b', expression.lower()))
    return len(fields | other_fields)


def calculate_alpha_score(
    sim_result: Dict[str, Any],
    prod_corr: float = 0.0,
    self_corr: float = 0.0,
    pool_corr: float = 0.0,
    diversity_bonus: float = 0.0,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate composite alpha score from simulation results.
    
    Enhanced scoring formula:
    Score = w_test * S_test + w_train * S_train + w_fitness * Fitness
            + w_diversity * diversity_bonus
            + w_test_pass * test_pass_ratio
            - w_prod_corr * max(0, prod_corr - 0.7)
            - w_pool_corr * max(0, pool_corr - 0.5)
            - w_turnover * turnover_penalty
            - w_invest * investability_penalty
    
    Args:
        sim_result: Simulation result dictionary from Brain API
        prod_corr: Maximum correlation with production alphas (0-1)
        self_corr: Self-correlation value (0-1)
        pool_corr: Correlation with existing alpha pool (0-1)
        diversity_bonus: Pre-computed diversity bonus (0-1)
        weights: Optional custom weights, defaults to optimized values
    
    Returns:
        Composite score (higher is better)
    """
    w = weights or DEFAULT_WEIGHTS
    
    # Extract metrics with safe defaults
    is_stats = _extract_is_stats(sim_result)
    os_stats = _extract_os_stats(sim_result)
    invest_constrained = _extract_investability_stats(sim_result)
    risk_neutral = _extract_risk_neutral_stats(sim_result)
    
    # Core performance metrics
    test_sharpe = _safe_float(os_stats.get('sharpe', os_stats.get('Sharpe', 0.0)))
    train_sharpe = _safe_float(is_stats.get('sharpe', is_stats.get('Sharpe', 0.0)))
    fitness = _safe_float(is_stats.get('fitness', is_stats.get('Fitness', 0.0)))
    
    # Risk/constraint metrics
    turnover = _safe_float(is_stats.get('turnover', is_stats.get('Turnover', 0.0)))
    invest_sharpe = _safe_float(invest_constrained.get('sharpe', invest_constrained.get('Sharpe', train_sharpe)))
    rn_sharpe = _safe_float(risk_neutral.get('sharpe', risk_neutral.get('Sharpe', train_sharpe)))
    
    # Test pass ratio
    test_results = evaluate_alpha_tests(sim_result)
    tests_passed = sum(1 for v in test_results.values() if v)
    tests_total = len(test_results)
    test_pass_ratio = tests_passed / tests_total if tests_total > 0 else 0.0
    
    # Calculate penalties
    prod_corr_penalty = max(0, prod_corr - CORRELATION_THRESHOLD)
    pool_corr_penalty = max(0, pool_corr - 0.5)  # Lower threshold for pool
    
    # Turnover penalty: penalize both extremes
    if turnover > TURNOVER_MAX:
        turnover_penalty = turnover - TURNOVER_MAX
    elif turnover < TURNOVER_MIN:
        turnover_penalty = TURNOVER_MIN - turnover
    else:
        turnover_penalty = 0.0
    
    # Investability penalty: difference between raw and constrained Sharpe
    investability_penalty = max(0, train_sharpe - invest_sharpe)
    
    # Risk adjustment bonus: reward alphas that improve under risk neutralization
    risk_bonus = max(0, rn_sharpe - train_sharpe) * 0.1
    
    # Calculate composite score
    score = (
        w.get('test_sharpe', 0.55) * test_sharpe +
        w.get('train_sharpe', 0.25) * train_sharpe +
        w.get('fitness', 0.20) * fitness +
        w.get('diversity_bonus', 0.15) * diversity_bonus +
        w.get('test_pass_bonus', 0.10) * test_pass_ratio +
        risk_bonus -
        w.get('prod_corr_penalty', 0.30) * prod_corr_penalty -
        w.get('pool_corr_penalty', 0.20) * pool_corr_penalty -
        w.get('turnover_penalty', 0.15) * turnover_penalty -
        w.get('investability_penalty', 0.20) * investability_penalty
    )
    
    logger.debug(
        f"得分明细: test={test_sharpe:.3f}, train={train_sharpe:.3f}, "
        f"fitness={fitness:.3f}, diversity={diversity_bonus:.3f}, "
        f"test_pass={test_pass_ratio:.2f}, prod_corr_pen={prod_corr_penalty:.3f}, "
        f"pool_corr_pen={pool_corr_penalty:.3f}, turnover_pen={turnover_penalty:.3f}, "
        f"invest_pen={investability_penalty:.3f} -> 总分 {score:.3f}"
    )
    
    return score


def calculate_rl_reward(
    sim_result: Dict[str, Any],
    alpha_pool: AlphaPool = None,
    expression: str = "",
    structure_sig: str = "",
    scale: float = 1.0
) -> Tuple[float, AlphaMetrics]:
    """
    Calculate reward for RL training with full metrics.
    
    Optimized for RL by:
    - Scaling rewards appropriately
    - Providing dense reward signal
    - Including diversity bonus
    
    Args:
        sim_result: Simulation result from Brain API
        alpha_pool: Pool for diversity checking
        expression: Alpha expression string
        structure_sig: Structural signature for similarity
        scale: Reward scaling factor
        
    Returns:
        (reward, metrics) tuple
    """
    metrics = extract_alpha_metrics(sim_result, expression)
    
    # Compute pool-based metrics
    if alpha_pool:
        metrics.pool_correlation = alpha_pool.compute_pool_correlation(
            expression, structure_sig
        )
        diversity_bonus = alpha_pool.get_diversity_bonus(expression, structure_sig)
    else:
        diversity_bonus = 0.5  # Default moderate diversity
    
    # Calculate base score
    base_score = calculate_alpha_score(
        sim_result,
        prod_corr=metrics.prod_correlation,
        pool_corr=metrics.pool_correlation,
        diversity_bonus=diversity_bonus
    )
    
    # Add submission readiness bonus
    if is_submission_ready(metrics):
        base_score += 1.0
        metrics.submission_ready = True
    
    # Scale reward
    reward = base_score * scale
    
    metrics.raw_score = base_score
    metrics.adjusted_score = reward
    
    return reward, metrics


def extract_alpha_metrics(
    sim_result: Dict[str, Any],
    expression: str = ""
) -> AlphaMetrics:
    """
    Extract all metrics from simulation result.
    
    Args:
        sim_result: Simulation result dictionary
        expression: Alpha expression string
        
    Returns:
        AlphaMetrics object with all extracted values
    """
    is_stats = _extract_is_stats(sim_result)
    os_stats = _extract_os_stats(sim_result)
    invest_stats = _extract_investability_stats(sim_result)
    risk_stats = _extract_risk_neutral_stats(sim_result)
    
    # Extract test results
    test_results = evaluate_alpha_tests(sim_result)
    failed = [name for name, passed in test_results.items() if not passed]
    
    metrics = AlphaMetrics(
        alpha_id=sim_result.get('alpha_id', ''),
        expression=expression,
        train_sharpe=_safe_float(is_stats.get('sharpe', is_stats.get('Sharpe', 0.0))),
        test_sharpe=_safe_float(os_stats.get('sharpe', os_stats.get('Sharpe', 0.0))),
        train_fitness=_safe_float(is_stats.get('fitness', is_stats.get('Fitness', 0.0))),
        test_fitness=_safe_float(os_stats.get('fitness', os_stats.get('Fitness', 0.0))),
        turnover=_safe_float(is_stats.get('turnover', is_stats.get('Turnover', 0.0))),
        drawdown=_safe_float(is_stats.get('drawdown', is_stats.get('Drawdown', 0.0))),
        risk_neutral_sharpe=_safe_float(risk_stats.get('sharpe', risk_stats.get('Sharpe', 0.0))),
        invest_constrained_sharpe=_safe_float(invest_stats.get('sharpe', invest_stats.get('Sharpe', 0.0))),
        tests_passed=len(test_results) - len(failed),
        tests_total=len(test_results),
        failed_tests=failed
    )
    
    return metrics


def is_submission_ready(metrics: AlphaMetrics) -> bool:
    """
    Check if alpha meets Brain submission criteria.
    
    Criteria:
    - Sharpe >= 1.58
    - Fitness >= 1.0
    - Pass all critical tests
    - Reasonable turnover
    - Low correlation with production
    """
    # Core performance thresholds
    if metrics.train_sharpe < SHARPE_THRESHOLD:
        return False
    
    if metrics.train_fitness < FITNESS_THRESHOLD:
        return False
    
    # Turnover range
    if metrics.turnover > TURNOVER_MAX or metrics.turnover < TURNOVER_MIN:
        return False
    
    # Correlation threshold
    if metrics.prod_correlation > CORRELATION_THRESHOLD:
        return False
    
    # Critical test failures
    critical_tests = ['LOW_SHARPE', 'LOW_FITNESS', 'CONCENTRATED_WEIGHT', 
                     'HIGH_TURNOVER', 'SIMILAR_TO_PRODUCTION']
    for test in critical_tests:
        if test in metrics.failed_tests:
            return False
    
    return True


def compute_marginal_contribution(
    new_alpha_score: float,
    pool_stats: Dict,
    correlation: float
) -> float:
    """
    Compute marginal contribution of new alpha to pool (AlphaGen style).
    
    Marginal contribution = alpha_score * (1 - correlation_with_pool)
    
    This encourages alphas that are both good AND diverse.
    
    Args:
        new_alpha_score: Score of the new alpha
        pool_stats: Statistics of existing pool
        correlation: Max correlation with pool
        
    Returns:
        Marginal contribution value
    """
    diversity_factor = 1.0 - correlation
    return new_alpha_score * diversity_factor


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_is_stats(sim_result: Dict) -> Dict:
    """从模拟结果中提取训练集统计信息。"""
    if 'train' in sim_result and sim_result['train']:
        return sim_result['train']
    if 'is_stats' in sim_result:
        is_stats = sim_result['is_stats']
        if isinstance(is_stats, list) and len(is_stats) > 0:
            return is_stats[0]
    if 'is' in sim_result:
        return sim_result['is'] or {}
    if 'pnl' in sim_result and isinstance(sim_result['pnl'], dict):
        return sim_result['pnl'].get('is', {}) or {}
    return {}


def _extract_os_stats(sim_result: Dict) -> Dict:
    """从模拟结果中提取测试集统计信息。"""
    if 'test' in sim_result and sim_result['test']:
        return sim_result['test']
    if 'os' in sim_result:
        return sim_result['os'] or {}
    if 'pnl' in sim_result and isinstance(sim_result['pnl'], dict):
        return sim_result['pnl'].get('os', {}) or {}
    return {}


def _extract_investability_stats(sim_result: Dict) -> Dict:
    """提取可投资性受限的统计信息。"""
    train_stats = _extract_is_stats(sim_result)
    if 'investabilityConstrained' in train_stats:
        return train_stats['investabilityConstrained'] or {}
    if 'investabilityConstrained' in sim_result:
        return sim_result['investabilityConstrained'] or {}
    return {}


def _extract_risk_neutral_stats(sim_result: Dict) -> Dict:
    """提取风险中性化后的统计信息。"""
    if 'riskNeutralized' in sim_result:
        return sim_result['riskNeutralized'] or {}
    train_stats = _extract_is_stats(sim_result)
    if 'riskNeutralized' in train_stats:
        return train_stats['riskNeutralized'] or {}
    return {}


def _safe_float(value: Any) -> float:
    """安全地转换为浮点数，失败则返回 0.0。"""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def evaluate_alpha_tests(sim_result: Dict) -> Dict[str, bool]:
    """
    评估 Alpha 通过了哪些测试。
    
    Returns:
        字典: 测试名称: 是否通过 (True/False)
    """
    tests = sim_result.get('tests', {})
    if not tests:
        # Try is_tests
        is_tests = sim_result.get('is_tests', [])
        if isinstance(is_tests, list):
            tests = {t.get('name', ''): t.get('result', 'FAIL') for t in is_tests if isinstance(t, dict)}
    
    if not tests:
        return {}
    
    results = {}
    for test_name, test_result in tests.items():
        if isinstance(test_result, dict):
            results[test_name] = test_result.get('result') == 'PASS'
        elif isinstance(test_result, str):
            results[test_name] = test_result == 'PASS'
        else:
            results[test_name] = bool(test_result)
    
    return results


def get_failed_tests(sim_result: Dict) -> list:
    """获取 Alpha 未通过的测试列表。"""
    test_results = evaluate_alpha_tests(sim_result)
    return [name for name, passed in test_results.items() if not passed]


def should_optimize(sim_result: Dict) -> Tuple[bool, str]:
    """
    Determine if an alpha should be optimized.
    
    Decision tree based on 优化.md recommendations.
    
    Returns:
        (should_optimize, reason) tuple
    """
    is_stats = _extract_is_stats(sim_result) or {}
    os_stats = _extract_os_stats(sim_result) or {}
    invest_stats = _extract_investability_stats(sim_result) or {}
    risk_stats = _extract_risk_neutral_stats(sim_result) or {}

    train_sharpe = _safe_float(is_stats.get('sharpe', is_stats.get('Sharpe', 0)))
    train_fitness = _safe_float(is_stats.get('fitness', is_stats.get('Fitness', 0)))
    train_turnover = _safe_float(is_stats.get('turnover', is_stats.get('Turnover', 0)))

    test_sharpe = _safe_float(os_stats.get('sharpe', os_stats.get('Sharpe', 0)))
    test_fitness = _safe_float(os_stats.get('fitness', os_stats.get('Fitness', 0)))

    invest_sharpe = _safe_float(invest_stats.get('sharpe', invest_stats.get('Sharpe', train_sharpe)))
    rn_sharpe = _safe_float(risk_stats.get('sharpe', risk_stats.get('Sharpe', train_sharpe)))

    # ---- 0) Fast reject: clearly bad / noisy ----
    if train_sharpe <= 0 and test_sharpe <= 0:
        return False, "IS/OOS均为负，淘汰"

    if train_sharpe < 0.15 and rn_sharpe < 0.4:
        return False, "信号过弱且风险中性化未救回，淘汰"

    # ---- 1) Already good ----
    if train_sharpe >= SHARPE_THRESHOLD and train_fitness >= FITNESS_THRESHOLD:
        if test_sharpe >= 0.8:
            return False, "已接近/达到门槛且OOS不差，跳过优化"
        return True, "IS达标但OOS偏弱，做稳健性优化"

    # ---- 2) High-value optimization triggers ----
    # A) Risk exposure issue
    if (rn_sharpe - train_sharpe) >= 0.25 and rn_sharpe >= 0.6:
        return True, "风险中性化显著改善：优先调neutralization/结构去风险"

    # B) Investability drops
    if (train_sharpe - invest_sharpe) >= 0.25 and train_sharpe >= 0.3:
        return True, "可投资性约束下掉得多：优先降集中/做更强归一化/更平滑"

    # C) Overfitting
    if train_sharpe >= 0.4:
        ratio = test_sharpe / (train_sharpe + 1e-9)
        gap = train_sharpe - test_sharpe
        if ratio < 0.5 and gap >= 0.3:
            return True, "IS→OOS衰减明显：优先加平滑/增大窗口/提高decay"

    # D) Turnover too extreme
    if train_turnover > TURNOVER_MAX:
        return True, "换手过高：优先增大窗口/加decay/改更平滑结构"

    # ---- 3) Sweet spot: positive but not yet passing ----
    if 0.15 <= train_sharpe < SHARPE_THRESHOLD:
        if test_sharpe > -0.2 and test_fitness > -0.2:
            return True, "正信号但未达标：做窗口/标准化/settings小扫"
        return False, "OOS过差且无救回迹象，淘汰"

    # ---- 4) Default ----
    return True, "默认：可尝试低成本优化"


def rank_alphas(
    alphas: List[Dict],
    alpha_pool: AlphaPool = None
) -> List[Tuple[Dict, float, AlphaMetrics]]:
    """
    Rank a list of alphas by score with diversity consideration.
    
    Args:
        alphas: List of alpha dicts with simulation results
        alpha_pool: Optional pool for diversity checking
        
    Returns:
        List of (alpha_dict, score, metrics) sorted by score descending
    """
    ranked = []
    
    for alpha in alphas:
        sim_result = alpha.get('simulation', {})
        expression = alpha.get('alpha_expression', alpha.get('expression', ''))
        
        if not sim_result:
            continue
        
        # Compute metrics
        metrics = extract_alpha_metrics(sim_result, expression)
        
        # Compute diversity bonus
        diversity = 0.5  # Default
        if alpha_pool:
            diversity = alpha_pool.get_diversity_bonus(expression)
            metrics.pool_correlation = alpha_pool.compute_pool_correlation(expression)
        
        # Calculate score
        score = calculate_alpha_score(
            sim_result,
            pool_corr=metrics.pool_correlation,
            diversity_bonus=diversity
        )
        
        metrics.raw_score = score
        ranked.append((alpha, score, metrics))
    
    # Sort by score descending
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    return ranked
