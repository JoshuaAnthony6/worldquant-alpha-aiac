"""
Optimization Chain Module

Implements the "Factor Optimization Chain" from Chain-of-Alpha methodology.
Takes weak/failed alphas + backtest feedback and generates local rewrites
to iteratively improve performance.

Enhanced with:
- Systematic window parameter sweeps
- Settings grid search (neutralization, decay, truncation)
- Structure mutations (sign flip, wrapper add/remove, operator swap)
- LLM-guided rewrites with backtest feedback

Reference: 优化.md Section 3.3, 3.4, Chain-of-Alpha (2025)
"""

import re
import os
import json
from typing import List, Dict, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass, field
from itertools import product

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Constants
# ============================================================================

# Window values for parameter sweep (expanded range)
WINDOW_OPTIONS = [5, 10, 20, 22, 40, 60, 66, 120, 126, 252, 500]

# Short and long windows for relative strategies
SHORT_WINDOWS = [5, 10, 20]
LONG_WINDOWS = [60, 120, 252]

# Decay values for settings sweep  
DECAY_OPTIONS = [0, 2, 4, 8, 12, 16]

# Neutralization options (ordered by aggressiveness)
NEUTRALIZATION_OPTIONS = [
    "MARKET",       # Least aggressive
    "SECTOR", 
    "INDUSTRY", 
    "SUBINDUSTRY",  # Most aggressive
    "CROWDING",     # Special: reduce crowding
    "FAST"          # Special: faster decay
]

# Truncation options
TRUNCATION_OPTIONS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]

# Wrapper functions that can be added/removed
WRAPPER_FUNCTIONS = ['rank', 'scale', 'zscore', 'quantile']

# Time-series functions that typically have window parameters
TS_WINDOW_FUNCTIONS = [
    'ts_mean', 'ts_delta', 'ts_zscore', 'ts_std_dev', 
    'ts_rank', 'ts_ir', 'ts_returns', 'ts_decay_linear',
    'ts_sum', 'ts_min', 'ts_max', 'ts_argmax', 'ts_argmin',
    'ts_corr', 'ts_cov', 'ts_skewness', 'ts_kurtosis'
]


@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    original_expression: str
    optimized_expression: str
    change_type: str
    change_description: str
    original_score: float = 0.0
    optimized_score: float = 0.0
    improvement: float = 0.0
    settings: Dict = field(default_factory=dict)
    simulation_result: Dict = field(default_factory=dict)


# ============================================================================
# Local Rewrite Generation
# ============================================================================

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
    
    # 1. Sign flip variants (always try)
    variants.extend(_generate_sign_variants(expression))
    
    # 2. Window sweep variants
    variants.extend(_generate_window_variants(expression))
    
    # 3. Wrapper variants (rank, zscore, winsorize)
    variants.extend(_generate_wrapper_variants(expression))
    
    # 4. Operator substitution variants
    variants.extend(_generate_operator_substitutions(expression))
    
    # 5. Complexity reduction variants
    if _estimate_complexity(expression) > 5:
        variants.extend(_generate_simplification_variants(expression))
    
    # Prioritize based on feedback/failure mode
    variants = _prioritize_variants(variants, reason, failed_tests)
    
    # Remove duplicates
    seen = set()
    unique_variants = []
    for v in variants:
        if v['expression'] not in seen:
            seen.add(v['expression'])
            unique_variants.append(v)
    
    return unique_variants[:max_variants]


def _generate_sign_variants(expression: str) -> List[Dict]:
    """生成符号反转和单调变换变体。"""
    variants = []
    
    # Negative sign (reversal)
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
    
    # Signed square (amplify signal)
    variants.append({
        'expression': f"signed_power({expression}, 2)",
        'change_type': 'sign_transform',
        'description': '平方放大（保持符号）'
    })
    
    # Signed square root (dampen signal)
    variants.append({
        'expression': f"signed_power({expression}, 0.5)",
        'change_type': 'sign_transform', 
        'description': '平方根压缩（保持符号）'
    })
    
    return variants


def _generate_window_variants(expression: str) -> List[Dict]:
    """生成窗口参数扫描变体。"""
    variants = []
    
    # Find all numeric parameters that look like windows
    # Pattern: function_name(field, NUMBER) or function_name(field, NUMBER, ...)
    window_pattern = r'(\w+)\s*\(\s*([^,]+)\s*,\s*(\d+)'
    
    matches = list(re.finditer(window_pattern, expression))
    
    for match in matches:
        func_name = match.group(1)
        original_window = int(match.group(3))
        
        # Skip if not a typical window function
        if func_name not in TS_WINDOW_FUNCTIONS:
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
    
    # Also try adding smoothing via ts_mean if no ts_mean present
    if 'ts_mean' not in expression:
        for window in [5, 10, 20]:
            variants.append({
                'expression': f"ts_mean({expression}, {window})",
                'change_type': 'window_add',
                'description': f'添加 {window} 日平滑'
            })
    
    return variants


def _generate_wrapper_variants(expression: str) -> List[Dict]:
    """生成套壳函数变体（rank, zscore, winsorize）。"""
    variants = []
    
    # Detect existing wrapper
    has_rank = expression.startswith('rank(')
    has_zscore = expression.startswith('zscore(') or expression.startswith('ts_zscore(')
    has_scale = expression.startswith('scale(')
    
    # Remove wrapper if exists
    if has_rank:
        # Find matching parenthesis
        inner = _extract_inner_expression(expression, 'rank')
        if inner:
            variants.append({
                'expression': inner,
                'change_type': 'wrapper_remove',
                'description': '移除 rank() 套壳'
            })
            # Replace with zscore
            variants.append({
                'expression': f"ts_zscore({inner}, 60)",
                'change_type': 'wrapper_change',
                'description': '将 rank() 改为 ts_zscore(60)'
            })
            variants.append({
                'expression': f"zscore({inner})",
                'change_type': 'wrapper_change',
                'description': '将 rank() 改为 zscore()'
            })
    else:
        # Add rank wrapper
        variants.append({
            'expression': f"rank({expression})",
            'change_type': 'wrapper_add',
            'description': '添加 rank() 套壳'
        })
    
    if not has_zscore:
        # Add zscore variants with different windows
        for window in [20, 60, 120]:
            variants.append({
                'expression': f"ts_zscore({expression}, {window})",
                'change_type': 'wrapper_add',
                'description': f'添加 ts_zscore({window}) 套壳'
            })
    
    if not has_scale:
        variants.append({
            'expression': f"scale({expression})",
            'change_type': 'wrapper_add',
            'description': '添加 scale() 套壳'
        })
    
    # Winsorize for concentration issues (with different limits)
    for limit in [0.01, 0.02, 0.05]:
        variants.append({
            'expression': f"winsorize({expression}, std={limit})",
            'change_type': 'wrapper_add',
            'description': f'添加 winsorize(std={limit}) 以控制持仓集中度'
        })
    
    # Group neutralize
    for group in ['sector', 'industry', 'subindustry']:
        variants.append({
            'expression': f"group_neutralize({expression}, {group})",
            'change_type': 'wrapper_add',
            'description': f'添加 {group} 中性化'
        })
    
    return variants


def _generate_operator_substitutions(expression: str) -> List[Dict]:
    """生成算子替换变体。"""
    variants = []
    
    # Common substitution pairs
    substitutions = [
        ('ts_delta', 'ts_returns'),
        ('ts_returns', 'ts_delta'),
        ('ts_mean', 'ts_decay_linear'),
        ('ts_decay_linear', 'ts_mean'),
        ('ts_std_dev', 'ts_ir'),
        ('rank', 'quantile'),
    ]
    
    for old_op, new_op in substitutions:
        if old_op in expression:
            new_expr = expression.replace(old_op, new_op, 1)
            if new_expr != expression:
                variants.append({
                    'expression': new_expr,
                    'change_type': 'operator_substitute',
                    'description': f'将 {old_op} 替换为 {new_op}'
                })
    
    return variants


def _generate_simplification_variants(expression: str) -> List[Dict]:
    """生成简化变体，降低复杂度。"""
    variants = []
    
    # Try to simplify nested structures
    # Pattern: remove one level of nesting
    nested_pattern = r'(\w+)\(\s*(\w+)\(([^)]+)\)\s*\)'
    match = re.search(nested_pattern, expression)
    
    if match:
        outer_func = match.group(1)
        inner_func = match.group(2)
        inner_content = match.group(3)
        
        # Try keeping only inner function
        simplified = f"{inner_func}({inner_content})"
        variants.append({
            'expression': expression.replace(match.group(0), simplified),
            'change_type': 'simplify',
            'description': f'简化：移除外层 {outer_func}()'
        })
        
        # Try keeping only outer function
        simplified = f"{outer_func}({inner_content})"
        variants.append({
            'expression': expression.replace(match.group(0), simplified),
            'change_type': 'simplify',
            'description': f'简化：移除内层 {inner_func}()'
        })
    
    return variants


def _extract_inner_expression(expression: str, wrapper: str) -> Optional[str]:
    """Extract inner expression from wrapper function."""
    if not expression.startswith(f"{wrapper}("):
        return None
    
    # Count parentheses to find matching close
    depth = 0
    start = len(wrapper) + 1
    for i, c in enumerate(expression):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return expression[start:i]
    return None


def _estimate_complexity(expression: str) -> int:
    """Estimate expression complexity (number of operators)."""
    # Count function calls
    return len(re.findall(r'\w+\(', expression))


def _prioritize_variants(
    variants: List[Dict],
    reason: str,
    failed_tests: List[str]
) -> List[Dict]:
    """重新排序变体，基于失败原因优先考虑最可能有帮助的改动。"""
    
    # Assign priority scores based on failure mode
    for v in variants:
        v['priority'] = 0
        change_type = v['change_type']
        
        if "riskNeutralized" in reason or "风险" in reason:
            # Prioritize neutralization and wrapper changes
            if 'neutralize' in v.get('description', '').lower():
                v['priority'] += 10
            if change_type == 'wrapper_add' and 'zscore' in v.get('description', ''):
                v['priority'] += 8
        
        if "investability" in reason.lower() or "可投资性" in reason:
            # Prioritize winsorize and concentration fixes
            if 'winsorize' in v.get('description', '').lower():
                v['priority'] += 10
            if change_type == 'wrapper_add':
                v['priority'] += 5
        
        if "OOS" in reason or "过拟合" in reason or "IS→OOS" in reason:
            # Prioritize smoothing (larger windows, decay)
            if change_type == 'window_sweep':
                # Higher priority for larger windows
                if 'window' in v.get('description', '') and any(str(w) in v['description'] for w in [60, 120, 252]):
                    v['priority'] += 10
            if 'smooth' in v.get('description', '').lower() or 'decay' in v.get('description', '').lower():
                v['priority'] += 8
        
        if "换手" in reason or "turnover" in reason.lower():
            # Prioritize smoothing and larger windows
            if change_type == 'window_sweep' or change_type == 'window_add':
                v['priority'] += 8
            if 'decay' in v.get('description', '').lower():
                v['priority'] += 10
        
        # Penalize sign flips slightly (usually less effective)
        if change_type == 'sign_flip':
            v['priority'] -= 2
        
        # Check for specific test failures
        if 'CONCENTRATED_WEIGHT' in failed_tests:
            if 'winsorize' in v.get('description', '').lower():
                v['priority'] += 15
        
        if 'HIGH_TURNOVER' in failed_tests:
            if change_type in ['window_sweep', 'window_add']:
                v['priority'] += 12
    
    # Sort by priority descending
    variants.sort(key=lambda x: x.get('priority', 0), reverse=True)
    
    return variants


# ============================================================================
# Settings Grid Search
# ============================================================================

def generate_settings_variants(
    base_settings: Dict,
    full_grid: bool = False
) -> List[Dict]:
    """
    生成环境设置扫描变体（中性化、衰减、截断）。
    这些是在模拟时应用的，而不是表达式级别。
    
    Args:
        base_settings: Base simulation settings
        full_grid: If True, generate full grid; otherwise smart subset
    
    Returns:
        List of settings variant dictionaries
    """
    variants = []
    
    base_neut = base_settings.get('neutralization', 'INDUSTRY')
    base_decay = base_settings.get('decay', 4)
    base_trunc = base_settings.get('truncation', 0.02)
    
    if full_grid:
        # Full grid search
        for neut in NEUTRALIZATION_OPTIONS:
            for decay in DECAY_OPTIONS:
                for trunc in TRUNCATION_OPTIONS:
                    if (neut, decay, trunc) != (base_neut, base_decay, base_trunc):
                        variants.append({
                            'neutralization': neut,
                            'decay': decay,
                            'truncation': trunc,
                            'change_type': 'settings_full',
                            'description': f'neutralization={neut}, decay={decay}, truncation={trunc}'
                        })
    else:
        # Smart subset: vary one parameter at a time
        
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
        
        # Also add a few "diagonal" combinations that often help
        promising_combos = [
            ('SUBINDUSTRY', 8, 0.02),  # More neutral + more decay
            ('INDUSTRY', 12, 0.01),     # Standard neutral + high decay + tight truncation
            ('SECTOR', 4, 0.05),        # Less neutral + wider truncation
            ('MARKET', 16, 0.02),       # Minimal neutral + high decay
        ]
        
        for neut, decay, trunc in promising_combos:
            if (neut, decay, trunc) != (base_neut, base_decay, base_trunc):
                variants.append({
                    'neutralization': neut,
                    'decay': decay,
                    'truncation': trunc,
                    'change_type': 'settings_combo',
                    'description': f'组合: neut={neut}, decay={decay}, trunc={trunc}'
                })
    
    return variants


# ============================================================================
# LLM-Guided Optimization
# ============================================================================

def create_optimization_prompt(
    expression: str,
    sim_result: Dict,
    pool_corr: float = 0.0,
    previous_attempts: List[Dict] = None
) -> str:
    """
    创建基于 LLM 的优化建议提示词。
    
    Args:
        expression: Original alpha expression
        sim_result: Simulation result
        pool_corr: Correlation with existing pool
        previous_attempts: Previous optimization attempts and results
    """
    from alpha_scoring import get_failed_tests, should_optimize, _extract_is_stats, _extract_os_stats
    
    failed = get_failed_tests(sim_result)
    should_opt, reason = should_optimize(sim_result)
    
    # Extract key metrics
    is_stats = _extract_is_stats(sim_result) or {}
    os_stats = _extract_os_stats(sim_result) or {}
    risk_neutral = sim_result.get('riskNeutralized', {}) or {}
    invest = sim_result.get('investabilityConstrained', {}) or {}
    
    # Format previous attempts if any
    attempts_str = ""
    if previous_attempts:
        attempts_str = "\n**之前的优化尝试:**\n"
        for i, attempt in enumerate(previous_attempts[-5:], 1):  # Last 5 attempts
            attempts_str += f"{i}. 改动: {attempt.get('description', 'N/A')}\n"
            attempts_str += f"   结果: 分数 {attempt.get('score', 'N/A'):.3f}\n"
    
    prompt = f"""您正在优化一个需要改进的 Alpha 表达式。

**原始表达式:**
```
{expression}
```

**回测结果:**
- 训练集 Sharpe: {is_stats.get('sharpe', is_stats.get('Sharpe', 'N/A'))}
- 测试集 Sharpe: {os_stats.get('sharpe', os_stats.get('Sharpe', 'N/A'))}
- Fitness: {is_stats.get('fitness', is_stats.get('Fitness', 'N/A'))}
- 换手率 (Turnover): {is_stats.get('turnover', is_stats.get('Turnover', 'N/A'))}
- 风险中性化后的 Sharpe: {risk_neutral.get('sharpe', risk_neutral.get('Sharpe', 'N/A'))}
- 考虑可投资性限制后的 Sharpe: {invest.get('sharpe', invest.get('Sharpe', 'N/A'))}
- 与现有池的相关性: {pool_corr:.3f}

**未通过的测试:** {', '.join(failed) if failed else '无'}

**优化原因:** {reason}
{attempts_str}
**您的任务:**
生成 5-10 个可能改进此 Alpha 的局部改写方案。

**优化策略指南:**
1. **如果风险中性化后效果更好** (riskNeutralized Sharpe >> 原始 Sharpe):
   - 添加 group_neutralize(expr, sector/industry/subindustry)
   - 调整模拟设置的 neutralization 参数
   - 考虑去掉可能带来风险暴露的算子

2. **如果可投资性受限效果差** (investabilityConstrained Sharpe << 原始 Sharpe):
   - 添加 winsorize(expr, std=0.02) 降低集中度
   - 添加 scale() 或 rank() 做更均匀的权重分配
   - 减少极端值：使用 clip() 或更大的 truncation

3. **如果测试集 << 训练集 * 0.5** (过拟合):
   - 增大时间窗口参数 (如 20 -> 60 -> 120)
   - 添加 ts_decay_linear 平滑
   - 增加模拟设置的 decay 参数
   - 简化表达式结构

4. **如果换手率过高**:
   - 增大时间窗口
   - 添加 ts_mean() 平滑
   - 增加 decay 参数

5. **通用改进**:
   - 尝试 -expr (信号反转)
   - 尝试不同的归一化方式 (rank vs zscore)
   - 尝试相似但更稳健的算子替换

**输出格式:**
请以 JSON 列表格式输出您的优化建议:
```json
[
  {{
    "expression": "优化后的完整表达式",
    "rationale": "改动原因和预期效果"
  }}
]
```

**重要提醒:**
- 输出的表达式必须是完整的、可直接用于模拟的 WorldQuant Brain 表达式
- 不要生成与原始表达式完全相同的表达式
- 优先考虑针对上述优化原因的改动
"""
    return prompt


def llm_guided_rewrite(
    expression: str,
    sim_result: Dict,
    pool_corr: float = 0.0,
    previous_attempts: List[Dict] = None,
    llm_client = None,
    max_rewrites: int = 10
) -> List[Dict]:
    """
    Use LLM to generate optimization suggestions.
    
    Args:
        expression: Original expression
        sim_result: Simulation result
        pool_corr: Pool correlation
        previous_attempts: Previous attempts
        llm_client: OpenAI-compatible client
        max_rewrites: Maximum number of rewrites to generate
        
    Returns:
        List of rewrite dictionaries
    """
    if llm_client is None:
        try:
            from openai import OpenAI
            llm_client = OpenAI(
                base_url=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")
            return []
    
    prompt = create_optimization_prompt(
        expression, sim_result, pool_corr, previous_attempts
    )
    
    try:
        response = llm_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
            messages=[
                {"role": "system", "content": "你是一个专业的量化研究员，擅长优化 WorldQuant Brain alpha 表达式。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        
        # Clean and parse JSON
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        # Handle both list and object formats
        parsed = json.loads(content.strip())
        if isinstance(parsed, dict):
            rewrites = parsed.get('rewrites', parsed.get('alphas', [parsed]))
        else:
            rewrites = parsed
        
        # Convert to standard format
        variants = []
        for r in rewrites[:max_rewrites]:
            if isinstance(r, dict) and 'expression' in r:
                variants.append({
                    'expression': r['expression'],
                    'change_type': 'llm_rewrite',
                    'description': r.get('rationale', 'LLM 优化建议')
                })
        
        return variants
        
    except Exception as e:
        logger.error(f"LLM rewrite failed: {e}")
        return []


# ============================================================================
# Comprehensive Optimization Pipeline
# ============================================================================

def optimize_alpha(
    expression: str,
    sim_result: Dict,
    brain_session,
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    base_settings: Dict = None,
    max_variants: int = 50,
    max_eval: int = 20,
    use_llm: bool = True,
    score_function: Callable = None
) -> OptimizationResult:
    """
    Comprehensive alpha optimization pipeline.
    
    1. Generate local rewrites
    2. Generate settings variants
    3. Optionally get LLM suggestions
    4. Evaluate top candidates
    5. Return best result
    
    Args:
        expression: Original alpha expression
        sim_result: Original simulation result
        brain_session: Brain API session
        region: Market region
        universe: Stock universe
        delay: Data delay
        base_settings: Base simulation settings
        max_variants: Maximum variants to generate
        max_eval: Maximum variants to evaluate
        use_llm: Whether to use LLM for suggestions
        score_function: Custom scoring function
        
    Returns:
        OptimizationResult with best variant
    """
    import ace_lib as ace
    from alpha_scoring import calculate_alpha_score
    
    score_fn = score_function or calculate_alpha_score
    
    # Default settings
    base_settings = base_settings or {
        'neutralization': 'INDUSTRY',
        'decay': 4,
        'truncation': 0.02
    }
    
    # Calculate original score
    original_score = score_fn(sim_result)
    
    logger.info(f"Starting optimization for: {expression[:50]}...")
    logger.info(f"Original score: {original_score:.4f}")
    
    # Collect all candidates
    candidates = []
    
    # 1. Local rewrites (expression-level changes)
    local_rewrites = generate_local_rewrites(
        expression, sim_result, max_variants=max_variants // 2
    )
    for r in local_rewrites:
        candidates.append({
            'expression': r['expression'],
            'settings': base_settings.copy(),
            'change_type': r['change_type'],
            'description': r['description']
        })
    
    # 2. Settings variants (keeping original expression)
    settings_variants = generate_settings_variants(base_settings, full_grid=False)
    for s in settings_variants:
        candidates.append({
            'expression': expression,
            'settings': {
                'neutralization': s['neutralization'],
                'decay': s['decay'],
                'truncation': s['truncation']
            },
            'change_type': s['change_type'],
            'description': s['description']
        })
    
    # 3. LLM suggestions
    if use_llm:
        llm_rewrites = llm_guided_rewrite(expression, sim_result)
        for r in llm_rewrites:
            candidates.append({
                'expression': r['expression'],
                'settings': base_settings.copy(),
                'change_type': r['change_type'],
                'description': r['description']
            })
    
    logger.info(f"Generated {len(candidates)} optimization candidates")
    
    # 4. Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = (c['expression'], str(c['settings']))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    candidates = unique_candidates[:max_eval]
    logger.info(f"Evaluating {len(candidates)} unique candidates")
    
    # 5. Evaluate candidates
    best_result = OptimizationResult(
        original_expression=expression,
        optimized_expression=expression,
        change_type='none',
        change_description='无改进',
        original_score=original_score,
        optimized_score=original_score,
        improvement=0.0,
        settings=base_settings
    )
    
    # Prepare simulation configs
    configs = []
    for candidate in candidates:
        settings = candidate['settings']
        config = ace.generate_alpha(
            regular=candidate['expression'],
            alpha_type="REGULAR",
            region=region,
            universe=universe,
            delay=delay,
            neutralization=settings['neutralization'],
            decay=settings['decay'],
            truncation=settings['truncation'],
            pasteurization="ON",
            test_period="P2Y",
            unit_handling="VERIFY",
            nan_handling="ON",
            visualization=False
        )
        configs.append((config, candidate))
    
    # Batch simulate
    try:
        results = ace.simulate_alpha_list_multi(
            brain_session,
            [c[0] for c in configs],
            limit_of_concurrent_simulations=5,
            limit_of_multi_simulations=5
        )
        
        # Find best
        for i, (res, (_, candidate)) in enumerate(zip(results, configs)):
            if res.get('alpha_id'):
                score = score_fn(res)
                
                if score > best_result.optimized_score:
                    best_result = OptimizationResult(
                        original_expression=expression,
                        optimized_expression=candidate['expression'],
                        change_type=candidate['change_type'],
                        change_description=candidate['description'],
                        original_score=original_score,
                        optimized_score=score,
                        improvement=score - original_score,
                        settings=candidate['settings'],
                        simulation_result=res
                    )
                    logger.info(f"  New best: {score:.4f} ({candidate['description']})")
        
    except Exception as e:
        logger.error(f"Optimization evaluation failed: {e}")
    
    logger.info(f"Optimization complete. Improvement: {best_result.improvement:.4f}")
    
    return best_result


def batch_optimize(
    alphas: List[Dict],
    brain_session,
    region: str = "USA",
    universe: str = "TOP3000",
    max_per_alpha: int = 10,
    score_threshold: float = 0.0
) -> List[OptimizationResult]:
    """
    Batch optimize multiple alphas.
    
    Args:
        alphas: List of alpha dicts with expression and simulation
        brain_session: Brain API session
        region: Market region
        universe: Stock universe
        max_per_alpha: Maximum variants per alpha
        score_threshold: Minimum score to attempt optimization
        
    Returns:
        List of OptimizationResult objects
    """
    from alpha_scoring import should_optimize
    
    results = []
    
    for i, alpha in enumerate(alphas):
        expr = alpha.get('alpha_expression', alpha.get('expression', ''))
        sim = alpha.get('simulation', {})
        
        if not expr or not sim:
            continue
        
        # Check if worth optimizing
        should_opt, reason = should_optimize(sim)
        
        if not should_opt:
            logger.info(f"[{i+1}/{len(alphas)}] Skipping: {reason}")
            continue
        
        logger.info(f"[{i+1}/{len(alphas)}] Optimizing: {reason}")
        
        result = optimize_alpha(
            expression=expr,
            sim_result=sim,
            brain_session=brain_session,
            region=region,
            universe=universe,
            max_variants=max_per_alpha * 3,
            max_eval=max_per_alpha
        )
        
        results.append(result)
    
    return results
