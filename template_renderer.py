"""
Template Renderer Module

Converts structured JSON template specifications to valid Brain expression strings.
This enables controlled LLM output that is validated and rendered programmatically.

Reference: 优化.md Section 5 改法A
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Load template bank
TEMPLATE_BANK_PATH = Path(__file__).parent / "template_bank.json"


def load_template_bank() -> Dict:
    """Load the template bank from JSON file."""
    try:
        with open(TEMPLATE_BANK_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"在 {TEMPLATE_BANK_PATH} 未找到模板库文件")
        return {"templates": []}


def get_template_by_id(template_id: str) -> Optional[Dict]:
    """Get a template by its ID."""
    bank = load_template_bank()
    for template in bank.get("templates", []):
        if template.get("id") == template_id:
            return template
    return None


def list_templates(category: Optional[str] = None) -> List[Dict]:
    """List available templates, optionally filtered by category."""
    bank = load_template_bank()
    templates = bank.get("templates", [])
    
    if category:
        templates = [t for t in templates if t.get("category") == category]
    
    return templates


def render_expression(spec: Dict) -> str:
    """
    Render a Brain expression from a structured specification.
    
    Args:
        spec: Dictionary with keys:
            - template_id: ID of template to use
            - field / field1 / field2: Data field(s)
            - window / window1 / window2 / short_window / long_window: Lookback parameters
            - Any other template-specific parameters
    
    Returns:
        Rendered Brain expression string
    
    Example:
        spec = {
            "template_id": "momentum_rank",
            "field": "mdl110_growth",
            "window": 20
        }
        -> "rank(ts_delta(mdl110_growth, 20))"
    """
    template_id = spec.get("template_id")
    if not template_id:
        raise ValueError("spec must include 'template_id'")
    
    template = get_template_by_id(template_id)
    if not template:
        raise ValueError(f"Unknown template_id: {template_id}")
    
    template_str = template.get("template", "")
    
    # Build substitution dict from spec
    subs = {}
    
    # Standard field substitutions
    for key in ["field", "field1", "field2"]:
        if key in spec:
            subs[key] = spec[key]
    
    # Window substitutions
    for key in ["window", "window1", "window2", "short_window", "long_window", "n"]:
        if key in spec:
            subs[key] = str(spec[key])
    
    # Other parameters (power, limit, etc.)
    for key in ["power", "limit"]:
        if key in spec:
            subs[key] = str(spec[key])
    
    # Perform substitution
    try:
        expression = template_str.format(**subs)
        return expression
    except KeyError as e:
        raise ValueError(f"Missing required parameter for template '{template_id}': {e}")


def validate_spec(spec: Dict) -> tuple[bool, str]:
    """
    Validate a template specification before rendering.
    
    Returns:
        (is_valid, error_message)
    """
    template_id = spec.get("template_id")
    if not template_id:
        return False, "Missing 'template_id'"
    
    template = get_template_by_id(template_id)
    if not template:
        return False, f"Unknown template_id: {template_id}"
    
    template_str = template.get("template", "")
    
    # Check required fields
    required_fields = []
    if "{field}" in template_str:
        required_fields.append("field")
    if "{field1}" in template_str:
        required_fields.append("field1")
    if "{field2}" in template_str:
        required_fields.append("field2")
    
    for field in required_fields:
        if field not in spec:
            return False, f"Missing required field: {field}"
    
    # Check required windows
    window_params = ["window", "window1", "window2", "short_window", "long_window", "n"]
    for param in window_params:
        if "{" + param + "}" in template_str and param not in spec:
            return False, f"Missing required parameter: {param}"
    
    return True, ""


def batch_render(specs: List[Dict]) -> List[str]:
    """
    Render multiple expressions from a list of specifications.
    Skips invalid specs with a warning.
    """
    expressions = []
    
    for i, spec in enumerate(specs):
        is_valid, error = validate_spec(spec)
        if not is_valid:
            logger.warning(f"跳过无效规范 {i}: {error}")
            continue
        
        try:
            expr = render_expression(spec)
            expressions.append(expr)
        except Exception as e:
            logger.warning(f"渲染规范 {i} 失败: {e}")
    
    return expressions


def get_template_prompt_reference() -> str:
    """
    Generate a reference string for LLM prompts describing available templates.
    """
    bank = load_template_bank()
    templates = bank.get("templates", [])
    
    lines = ["可用于生成 Alpha 的模板："]
    lines.append("=" * 60)
    
    for t in templates:
        lines.append(f"ID: {t['id']}")
        lines.append(f"  类别: {t.get('category', '通用')}")
        lines.append(f"  模板: {t.get('template', '')}")
        lines.append(f"  描述: {t.get('description', '')}")
        if t.get('params'):
            lines.append(f"  参数: {t['params']}")
        lines.append("")
    
    return "\n".join(lines)


# Pydantic model for structured LLM output
try:
    from pydantic import BaseModel
    from typing import Optional, Union

    class AlphaSpec(BaseModel):
        """Structured specification for an alpha expression."""
        template_id: str
        field: Optional[str] = None
        field1: Optional[str] = None
        field2: Optional[str] = None
        window: Optional[int] = None
        window1: Optional[int] = None
        window2: Optional[int] = None
        short_window: Optional[int] = None
        long_window: Optional[int] = None
        power: Optional[float] = None
        limit: Optional[float] = None
        economic_rationale: str

    class AlphaSpecList(BaseModel):
        """List of alpha specifications from LLM."""
        alphas: List[AlphaSpec]

except ImportError:
    logger.warning("Pydantic 不可用，已禁用结构化输出验证")
    AlphaSpec = None
    AlphaSpecList = None
