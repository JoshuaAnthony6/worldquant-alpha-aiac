"""
Optimization Chain Module

Implements the "Factor Optimization Chain" from Chain-of-Alpha methodology.
Takes weak/failed alphas + backtest feedback and generates local rewrites
to iteratively improve performance.

Reference: 优化.md Section 3.3, 3.4
"""

import re
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Window values for parameter sweep
WINDOW_OPTIONS = [5, 22, 66, 126, 252, 500]

# Decay values for settings sweep  
DECAY_OPTIONS = [2, 8, 16]

# Neutralization options
NEUTRALIZATION_OPTIONS = ["MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY", "CROWDING", "FAST"]

# Truncation options
TRUNCATION_OPTIONS = [0.01, 0.02, 0.05, 0.08]


def generate_local_rewrites(
    expression: str,
    sim_result: Dict,
    feedback: Optional[str] = None,
    max_variants: int = 20
) -> List[Dict]:
    """
    Generate local rewrite variants for an alpha expression based on backtest feedback.
    
    Args:
        expression: Original alpha expression
        sim_result: Simulation result from Brain API
        feedback: Optional optimization feedback string
        max_variants: Maximum number of variants to generate
    
    Returns:
        List of variant dictionaries with 'expression', 'change_type', 'description'
    """
    from alpha_scoring import should_optimize, get_failed_tests
    
    variants = []
    
    # Determine optimization strategy based on feedback
    should_opt, reason = should_optimize(sim_result)
    failed_tests = get_failed_tests(sim_result)
    
    # 1. Sign flip variants
    variants.extend(_generate_sign_variants(expression))
    
    # 2. Window sweep variants
    variants.extend(_generate_window_variants(expression))
    
    # 3. Wrapper variants (rank, zscore, winsorize)
    variants.extend(_generate_wrapper_variants(expression))
    
    # 4. Settings sweep suggestions (returned as metadata, not expressions)
    # These are applied at simulation time, not expression level
    
    # Prioritize based on feedback
    if "riskNeutralized" in reason:
        # Prioritize neutralization variants
        variants = _prioritize_by_type(variants, "neutralization")
    elif "investability" in reason:
        # Prioritize winsorize/concentration fixes
        variants = _prioritize_by_type(variants, "wrapper")
    elif "test sharpe" in reason.lower():
        # Prioritize window adjustments (larger windows = more stable)
        variants = _prioritize_by_type(variants, "window")
    
    # Limit to max_variants
    return variants[:max_variants]


def _generate_sign_variants(expression: str) -> List[Dict]:
    """生成符号反转和单调变换变体。"""
    variants = []
    
    # Negative sign
    variants.append({
        'expression': f"-({expression})",
        'change_type': 'sign_flip',
        'description': '信号反转（Reversal）'
    })
    
    # Absolute value (removes direction, keeps magnitude)
    variants.append({
        'expression': f"abs({expression})",
        'change_type': 'sign_transform',
        'description': '绝对值（仅保留强度）'
    })
    
    return variants


def _generate_window_variants(expression: str) -> List[Dict]:
    """生成窗口参数扫描变体。"""
    variants = []
    
    # Find all numeric parameters that look like windows (typically after comma)
    # Pattern: function_name(field, NUMBER) or function_name(field, NUMBER, ...)
    window_pattern = r'(\w+)\s*\(\s*([^,]+)\s*,\s*(\d+)'
    
    matches = list(re.finditer(window_pattern, expression))
    
    for match in matches:
        func_name = match.group(1)
        original_window = int(match.group(3))
        
        # Skip if not a typical window function
        if func_name not in ['ts_mean', 'ts_delta', 'ts_zscore', 'ts_std_dev', 
                             'ts_rank', 'ts_ir', 'ts_returns', 'ts_decay_linear',
                             'ts_sum', 'ts_min', 'ts_max', 'ts_argmax', 'ts_argmin']:
            continue
        
        # Generate variants with different windows
        for new_window in WINDOW_OPTIONS:
            if new_window != original_window:
                new_expr = expression[:match.start(3)] + str(new_window) + expression[match.end(3):]
                variants.append({
                    'expression': new_expr,
                    'change_type': 'window_sweep',
                    'description': f'{func_name} 窗口: {original_window} -> {new_window}'
                })
    
    return variants


def _generate_wrapper_variants(expression: str) -> List[Dict]:
    """生成套壳函数变体（rank, zscore, winsorize）。"""
    variants = []
    
    # Already has rank? Try without or with zscore
    if expression.startswith('rank('):
        inner = expression[5:-1]  # Remove rank() wrapper
        variants.append({
            'expression': inner,
            'change_type': 'wrapper_remove',
            'description': '移除了 rank() 套壳'
        })
        variants.append({
            'expression': f"ts_zscore({inner}, 60)",
            'change_type': 'wrapper_change',
            'description': '将 rank() 改为 ts_zscore()'
        })
    else:
        # Add rank wrapper
        variants.append({
            'expression': f"rank({expression})",
            'change_type': 'wrapper_add',
            'description': '添加了 rank() 套壳'
        })
        # Add zscore wrapper
        variants.append({
            'expression': f"ts_zscore({expression}, 60)",
            'change_type': 'wrapper_add',
            'description': '添加了 ts_zscore() 套壳'
        })
    
    # Winsorize for concentration issues
    variants.append({
        'expression': f"winsorize({expression}, 0.02)",
        'change_type': 'wrapper_add',
        'description': '添加了 winsorize() 以控制持仓集中度'
    })
    
    return variants


def _prioritize_by_type(variants: List[Dict], priority_type: str) -> List[Dict]:
    """重新排序变体，优先考虑特定的更改类型。"""
    priority = [v for v in variants if v['change_type'].startswith(priority_type)]
    others = [v for v in variants if not v['change_type'].startswith(priority_type)]
    return priority + others


def generate_settings_variants(base_settings: Dict) -> List[Dict]:
    """
    生成环境设置扫描变体（中性化、衰减、截断）。
    这些是在模拟时应用的，而不是表达式级别。
    """
    variants = []
    
    base_neut = base_settings.get('neutralization', 'INDUSTRY')
    base_decay = base_settings.get('decay', 4)
    base_trunc = base_settings.get('truncation', 0.02)
    
    # Neutralization variants
    for neut in NEUTRALIZATION_OPTIONS:
        if neut != base_neut:
            variants.append({
                'neutralization': neut,
                'decay': base_decay,
                'truncation': base_trunc,
                'change_type': 'settings_neutralization',
                'description': f'中性化: {base_neut} -> {neut}'
            })
    
    # Decay variants
    for decay in DECAY_OPTIONS:
        if decay != base_decay:
            variants.append({
                'neutralization': base_neut,
                'decay': decay,
                'truncation': base_trunc,
                'change_type': 'settings_decay',
                'description': f'衰减: {base_decay} -> {decay}'
            })
    
    # Truncation variants
    for trunc in TRUNCATION_OPTIONS:
        if trunc != base_trunc:
            variants.append({
                'neutralization': base_neut,
                'decay': base_decay,
                'truncation': trunc,
                'change_type': 'settings_truncation',
                'description': f'截断: {base_trunc} -> {trunc}'
            })
    
    return variants


def create_optimization_prompt(
    expression: str,
    sim_result: Dict,
    pool_corr: float = 0.0
) -> str:
    """
    创建基于 LLM 的优化建议提示词。
    """
    from alpha_scoring import get_failed_tests, should_optimize
    
    failed = get_failed_tests(sim_result)
    should_opt, reason = should_optimize(sim_result)
    
    # Extract key metrics
    # Note: Using lowercase keys as per recent fix
    is_stats = sim_result.get('train', sim_result.get('is', {})) or {}
    if not is_stats and 'is_stats' in sim_result:
        is_stats = sim_result['is_stats'][0] if sim_result['is_stats'] else {}
    os_stats = sim_result.get('test', sim_result.get('os', {})) or {}
    risk_neutral = sim_result.get('riskNeutralized', {}) or {}
    invest = sim_result.get('investabilityConstrained', {}) or {}
    
    prompt = f"""您正在优化一个需要改进的 Alpha 表达式。
 
**原始表达式:**
```
{expression}
```
 
**回测结果:**
- 训练集 Sharpe: {is_stats.get('sharpe', 'N/A')}
- 测试集 Sharpe: {os_stats.get('sharpe', 'N/A')}
- Fitness: {is_stats.get('fitness', 'N/A')}
- 换手率 (Turnover): {is_stats.get('turnover', 'N/A')}
- 风险中性化后的 Sharpe: {risk_neutral.get('sharpe', 'N/A')}
- 考虑可投资性限制后的 Sharpe: {invest.get('sharpe', 'N/A')}
- 与现有池的相关性: {pool_corr:.3f}
 
**未通过的测试:** {', '.join(failed) if failed else '无'}
 
**优化原因:** {reason}
 
**您的任务:**
生成 5-10 个可能改进此 Alpha 的局部改写方案。侧重点：
1. 如果风险中性化后效果更好：通过中性化或结构调整减少风险暴露
2. 如果可投资性受限：添加 winsorize，减少持仓集中度
3. 如果测试集 < 训练集*0.5：增加衰减 (decay) 或窗口 (window) 以提高稳定性
4. 尝试不同的窗口参数
5. 如果动量可能反转，尝试符号反转
 
请以 JSON 列表格式输出:
[
  {{"expression": "...", "rationale": "..."}}
]
"""
    return prompt
