"""
Field Screening Module

Tests each data field with fixed templates to identify high-potential operands
before LLM generation. This significantly improves generation quality by
focusing on fields with proven information content.

Reference: 优化.md Section 3.2
"""

from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Fixed templates for field screening
SCREENING_TEMPLATES = [
    ("rank", "rank({field})"),
    ("rank_ts_mean", "rank(ts_mean({field}, {n}))"),
    ("rank_ts_delta", "rank(ts_delta({field}, {n}))"),
    ("rank_ts_rank", "rank(ts_rank({field}, {n}, constant=0))"),
    ("neg_rank_ts_delta", "-rank(ts_delta({field}, {n}))"),
    ("zscore", "ts_zscore({field}, {n})"),
]

# Standard lookback windows to test
SCREENING_WINDOWS = [5, 10, 20, 60, 120]


def screen_fields(
    session,
    dataset_id: str,
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    neutralization: str = "INDUSTRY",
    decay: int = 4,
    truncation: float = 0.02,
    top_k: int = 5,
    simulate_func=None,
    score_func=None,
) -> List[Dict]:
    """
    Screen data fields using fixed templates to find high-potential operands.
    
    Args:
        session: Brain API session
        dataset_id: Dataset to screen (e.g., "model110")  
        region: Market region
        universe: Stock universe
        delay: Data delay
        neutralization: Neutralization method
        decay: Decay value
        truncation: Truncation value
        top_k: Number of top fields to return
        simulate_func: Function to simulate alphas (from ace_lib)
        score_func: Function to calculate alpha score
    
    Returns:
        List of top_k fields with their scores and best templates
    """
    from alpha_scoring import calculate_alpha_score
    
    if score_func is None:
        score_func = calculate_alpha_score
    
    # Get available fields for the dataset
    fields = get_dataset_fields(session, dataset_id, region, universe, delay)
    
    if not fields:
        logger.warning(f"未找到数据集 {dataset_id} 的相关字段")
        return []
    
    print(f"\n{'='*80}")
    print(f"字段筛选：正在使用 {len(SCREENING_TEMPLATES)} 个模板测试 {len(fields)} 个字段")
    print(f"{'='*80}\n")
    
    field_scores = {}
    
    for field in tqdm(fields, desc="正在筛选字段"):
        field_id = field.get('id', field) if isinstance(field, dict) else field
        field_results = []
        
        for template_name, template in SCREENING_TEMPLATES:
            # For templates without window parameter
            if '{n}' not in template:
                expr = template.format(field=field_id)
                result = _test_expression(
                    session, expr, region, universe, delay,
                    neutralization, decay, truncation, simulate_func
                )
                if result:
                    score = score_func(result)
                    field_results.append({
                        'template': template_name,
                        'window': None,
                        'expression': expr,
                        'score': score,
                        'result': result
                    })
            else:
                # Test with different windows
                for window in SCREENING_WINDOWS:
                    expr = template.format(field=field_id, n=window)
                    result = _test_expression(
                        session, expr, region, universe, delay,
                        neutralization, decay, truncation, simulate_func
                    )
                    if result:
                        score = score_func(result)
                        field_results.append({
                            'template': template_name,
                            'window': window,
                            'expression': expr,
                            'score': score,
                            'result': result
                        })
        
        if field_results:
            # Find best result for this field
            best = max(field_results, key=lambda x: x['score'])
            field_scores[field_id] = {
                'field': field_id,
                'best_score': best['score'],
                'best_template': best['template'],
                'best_window': best['window'],
                'best_expression': best['expression'],
                'all_results': field_results
            }
    
    # Sort fields by best score
    sorted_fields = sorted(
        field_scores.values(),
        key=lambda x: x['best_score'],
        reverse=True
    )
    
    # Print top fields
    print(f"\n{'='*80}")
    print(f"得分最高的前 {top_k} 个字段：")
    print(f"{'='*80}")
    for i, f in enumerate(sorted_fields[:top_k], 1):
        print(f"  {i}. {f['field']}: 得分={f['best_score']:.4f} "
              f"(模板={f['best_template']}, 窗口={f['best_window']})")
    
    return sorted_fields[:top_k]


def _test_expression(
    session,
    expression: str,
    region: str,
    universe: str,
    delay: int,
    neutralization: str,
    decay: int,
    truncation: float,
    simulate_func=None
) -> Optional[Dict]:
    """
    Test a single expression and return simulation result.
    """
    try:
        # Use ace_lib's simulation function if provided
        if simulate_func:
            import ace_lib as ace
            alpha = ace.generate_alpha(
                regular_code=expression,
                region=region,
                universe=universe,
                delay=delay,
                neutralization=neutralization,
                decay=decay,
                truncation=truncation
            )
            # Simulate single alpha
            results = simulate_func(session, [alpha])
            if results and len(results) > 0:
                result = results[0]
                if result.get('status') == 'SUCCESS':
                    return result
        return None
    except Exception as e:
        logger.debug(f"表达式测试失败: {expression} - {e}")
        return None


def get_dataset_fields(
    session,
    dataset_id: str,
    region: str,
    universe: str,
    delay: int
) -> List[Dict]:
    """
    Get available fields for a dataset from Brain API.
    """
    try:
        import ace_lib as ace
        fields = ace.get_datafields(
            session,
            region=region,
            universe=universe,
            delay=delay,
            search=dataset_id
        )
        # Filter to only fields from the target dataset
        return [f for f in fields if f.get('id', '').startswith(dataset_id)]
    except Exception as e:
        logger.error(f"获取数据集字段时出错: {e}")
        return []


def create_field_whitelist(top_fields: List[Dict]) -> str:
    """
    Create a whitelist string for LLM prompt from top fields.
    """
    if not top_fields:
        return ""
    
    lines = ["推荐的高潜力数据字段（优先使用这些）："]
    for f in top_fields:
        lines.append(f"  - {f['field']} (最佳模板: {f['best_template']}, 得分: {f['best_score']:.3f})")
    
    return "\n".join(lines)
