"""
Alpha Scoring Module

Implements multi-objective scoring for alpha evaluation based on Brain simulation results.
Replaces single-metric (Sharpe) evaluation with a composite score that reflects
Brain's actual pass/fail criteria.

Reference: 优化.md Section 3.1
"""

from typing import Dict, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_alpha_score(
    sim_result: Dict[str, Any],
    prod_corr: float = 0.0,
    self_corr: float = 0.0,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate composite alpha score from simulation results.
    
    Score = w_test * S_test + w_train * S_train + w_fitness * Fitness
            - w_corr * max(0, prod_corr - 0.7)
            - w_turnover * turnover_penalty
            - w_invest * investability_penalty
    
    Args:
        sim_result: Simulation result dictionary from Brain API
        prod_corr: Maximum correlation with production alphas (0-1)
        self_corr: Self-correlation value (0-1)
        weights: Optional custom weights, defaults to optimized values
    
    Returns:
        Composite score (higher is better)
    """
    # Default weights based on 优化.md recommendations
    default_weights = {
        'test_sharpe': 0.55,
        'train_sharpe': 0.25,
        'fitness': 0.20,
        'prod_corr_penalty': 0.30,
        'turnover_penalty': 0.15,
        'investability_penalty': 0.20,
    }
    w = weights or default_weights
    
    # Extract metrics with safe defaults
    is_stats = _extract_is_stats(sim_result)
    os_stats = _extract_os_stats(sim_result)
    
    # Core performance metrics (note: lowercase keys in actual data)
    test_sharpe = os_stats.get('sharpe', os_stats.get('Sharpe', 0.0))
    train_sharpe = is_stats.get('sharpe', is_stats.get('Sharpe', 0.0))
    fitness = is_stats.get('fitness', is_stats.get('Fitness', 0.0))
    
    # Risk/constraint metrics
    turnover = is_stats.get('turnover', is_stats.get('Turnover', 0.0))
    
    # Investability-constrained metrics
    invest_constrained = _extract_investability_stats(sim_result)
    invest_sharpe = invest_constrained.get('sharpe', invest_constrained.get('Sharpe', train_sharpe))
    
    # Calculate penalties
    corr_penalty = max(0, prod_corr - 0.7)
    
    # Turnover penalty: penalize high turnover (> 50%)
    turnover_penalty = max(0, turnover - 0.5) if turnover else 0.0
    
    # Investability penalty: difference between raw and constrained Sharpe
    investability_penalty = max(0, train_sharpe - invest_sharpe) if invest_sharpe else 0.0
    
    # Calculate composite score
    score = (
        w['test_sharpe'] * _safe_float(test_sharpe) +
        w['train_sharpe'] * _safe_float(train_sharpe) +
        w['fitness'] * _safe_float(fitness) -
        w['prod_corr_penalty'] * corr_penalty -
        w['turnover_penalty'] * turnover_penalty -
        w['investability_penalty'] * investability_penalty
    )
    
    logger.debug(
        f"得分明细: 测试集={test_sharpe:.3f}, 训练集={train_sharpe:.3f}, "
        f"Fitness={fitness:.3f}, 相关性惩罚={corr_penalty:.3f}, "
        f"换手惩罚={turnover_penalty:.3f}, 可投资性惩罚={investability_penalty:.3f} -> 总分 {score:.3f}"
    )
    
    return score


def _extract_is_stats(sim_result: Dict) -> Dict:
    """从模拟结果中提取训练集统计信息。"""
    # Try multiple possible locations based on actual ace_lib output
    # Priority: train -> is_stats[0] -> is
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
    # Priority: test -> os
    if 'test' in sim_result and sim_result['test']:
        return sim_result['test']
    if 'os' in sim_result:
        return sim_result['os'] or {}
    if 'pnl' in sim_result and isinstance(sim_result['pnl'], dict):
        return sim_result['pnl'].get('os', {}) or {}
    return {}


def _extract_investability_stats(sim_result: Dict) -> Dict:
    """提取可投资性受限的统计信息。"""
    # Check in train stats first
    train_stats = _extract_is_stats(sim_result)
    if 'investabilityConstrained' in train_stats:
        return train_stats['investabilityConstrained'] or {}
    # Then check top-level
    if 'investabilityConstrained' in sim_result:
        return sim_result['investabilityConstrained'] or {}
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
    is_stats = _extract_is_stats(sim_result) or {}
    os_stats = _extract_os_stats(sim_result) or {}
    invest_stats = _extract_investability_stats(sim_result) or {}

    train_sharpe = _safe_float(is_stats.get('sharpe', is_stats.get('Sharpe', 0)))
    train_fitness = _safe_float(is_stats.get('fitness', is_stats.get('Fitness', 0)))
    train_turnover = _safe_float(is_stats.get('turnover', is_stats.get('Turnover', 0)))

    test_sharpe = _safe_float(os_stats.get('sharpe', os_stats.get('Sharpe', 0)))
    test_fitness = _safe_float(os_stats.get('fitness', os_stats.get('Fitness', 0)))

    invest_sharpe = _safe_float(invest_stats.get('sharpe', invest_stats.get('Sharpe', train_sharpe)))

    risk_neutral = sim_result.get('riskNeutralized', {}) or {}
    rn_sharpe = _safe_float(risk_neutral.get('sharpe', risk_neutral.get('Sharpe', train_sharpe)))

    # ---- 0) Fast reject: clearly bad / noisy ----
    # Negative in both IS and OOS: usually not worth 100-budget optimization
    if train_sharpe <= 0 and test_sharpe <= 0:
        return False, "IS/OOS均为负，淘汰"

    # Very weak + not rescued by RN: low ROI to optimize
    if train_sharpe < 0.15 and rn_sharpe < 0.4:
        return False, "信号过弱且风险中性化未救回，淘汰"

    # ---- 1) Already good (prefer tests-based if you have it) ----
    # If you can access pass/fail tests, check them here instead of hardcoding.
    if train_sharpe >= 1.58 and train_fitness >= 1.0:
        # still sanity-check OOS
        if test_sharpe >= 0.8:
            return False, "已接近/达到门槛且OOS不差，跳过优化"
        # else: good IS but weak OOS -> optimize for robustness
        return True, "IS达标但OOS偏弱，做稳健性优化"

    # ---- 2) High-value optimization triggers (fixable failure modes) ----
    # A) Risk exposure issue: RN improves a lot
    if (rn_sharpe - train_sharpe) >= 0.25 and rn_sharpe >= 0.6:
        return True, "风险中性化显著改善：优先调neutralization/结构去风险"

    # B) Investability drops a lot
    if (train_sharpe - invest_sharpe) >= 0.25 and train_sharpe >= 0.3:
        return True, "可投资性约束下掉得多：优先降集中/做更强归一化/更平滑"

    # C) Overfitting: big IS→OOS gap
    if train_sharpe >= 0.4:
        ratio = test_sharpe / (train_sharpe + 1e-9)
        gap = train_sharpe - test_sharpe
        if ratio < 0.5 and gap >= 0.3:
            return True, "IS→OOS衰减明显：优先加平滑/增大窗口/提高decay"

    # D) Turnover too extreme (if you have a target band)
    # Here we only trigger if it's extremely high/low; avoid over-filtering.
    if train_turnover > 0.6:
        return True, "换手过高：优先增大窗口/加decay/改更平滑结构"

    # ---- 3) The sweet spot: positive but not yet passing ----
    # This is where optimization pays off the most.
    if 0.15 <= train_sharpe < 1.58:
        # If OOS isn't catastrophic, worth optimizing
        if test_sharpe > -0.2 and test_fitness > -0.2:
            return True, "正信号但未达标：做窗口/标准化/settings小扫"
        # If OOS very bad, only optimize if RN rescues (already handled above)
        return False, "OOS过差且无救回迹象，淘汰"

    # ---- 4) Default ----
    return True, "默认：可尝试低成本优化"

