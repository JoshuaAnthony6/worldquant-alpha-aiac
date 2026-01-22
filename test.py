import os
import json
import re
import ace_lib as ace

# Optimization modules (优化.md recommendations)
from field_screening import screen_fields, create_field_whitelist
from template_renderer import get_template_prompt_reference, render_expression, AlphaSpecList
from optimization_chain import generate_local_rewrites, generate_settings_variants
from alpha_scoring import calculate_alpha_score, should_optimize

s = ace.start_session()

if ace.check_session_timeout(s) < 3000:
    s = ace.check_session_and_relogin(s)

def get_operators_reference(brain_session) -> str:
    operators_df = ace.get_operators(brain_session)
    operators_df = operators_df[operators_df['scope']=='REGULAR']
    operators_df = operators_df[['name', 'category', 'definition', 'description']]
    operators_df['description'] = operators_df['description'].apply(lambda x : x.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ') if isinstance(x, str) else x)
    operators_table = operators_df.to_markdown(index=False, tablefmt='github')
    operators_table_description = """## WorldQuant BRAIN 算子参考手册
 
### 表格结构
- `name`: 算子或函数名称（例如：`abs`, `add`, `log`）。
- `category`: 根据功能进行的分类（例如：算术运算、逻辑运算、时间序列等）。
- `definition`: 简短的语法或公式说明。
- `description`: 算子功能的简要说明，包括其用途和行为。
 
### 分类说明
1. 算术运算 (Arithmetic): 用于数学运算，如加、减、乘、除、对数和指数。
 
2. 逻辑运算 (Logical): 用于逻辑比较和条件判断，如 AND, OR, NOT 和相等检查。
 
3. 时间序列 (Time Series): 用于分析和操作时间序列数据，如计算指定天数内的移动平均、延迟或相关性。
 
4. 横截面 (Cross Sectional): 在单一时间点对多个仪器或实体进行操作，如排名 (ranking)、缩放 (scaling) 和归一化。
 
5. 向量 (Vector): 基于向量的计算，如查找向量中元素的总和、平均值或标准差。
 
6. 转换 (Transformational): 数据转换操作，如根据条件对值进行过滤、限制或尾部处理。
 
7. 分组 (Group): 基于分组的计算，如在中性化、排名或缩放组内（如行业或部门）的值。
 
8. 特殊 (Special): 用于特定用例，如单位换算或盈亏计算。
 
"""
    return operators_table_description + "\n" + operators_table

# Example output
print(get_operators_reference(s))

def get_dataset_reference(brain_session, dataset_id, region='USA', universe='TOP3000') -> str:
    datasets_df = ace.get_datasets(brain_session, region=region, universe=universe)
    dataset = datasets_df[datasets_df['id'] == dataset_id].iloc[0]
    dataset_details = f"""
## 数据集详情
 
ID: {dataset['id']}
名称: {dataset['name']}
分类名称: {dataset['category_name']}
子分类名称: {dataset['subcategory_name']}
说明: {dataset['description']}
 
以下是可用数据字段的参考表：
"""
    datafields_df = ace.get_datafields(brain_session, dataset_id=dataset_id, region=region, universe=universe)
    datafields_df = datafields_df[['id', 'description', 'type', 'dateCoverage', 'coverage', 'alphaCount']] # feel free to add or remove columns
    datafields_df['description'] = datafields_df['description'].apply(lambda x : x.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ') if isinstance(x, str) else x)
    datafields_table = datafields_df.to_markdown(index=False, tablefmt='github')

    return dataset_details + "\n" + datafields_table

# Example output
print(get_dataset_reference(s, "model110"))

from typing import List, Dict
from pydantic import BaseModel, Field


class AlphaExpression(BaseModel):
    """Structured output for a single alpha expression"""
    alpha_expression: str = Field(description="Alpha expression in WorldQuant Brain syntax")
    economic_rationale: str = Field(description="Economic reasoning behind the alpha expression")
    data_fields_used: List[str] = Field(description="List of data fields/features used in the expression")
    operators_used: List[str] = Field(description="List of operators used in the expression")


class AlphaExpressions(BaseModel):
    """Collection of alpha expressions"""
    alphas: List[AlphaExpression] = Field(description="List of generated alpha expressions")

from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # or os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # or os.getenv("OPENAI_BASE_URL") 

def clean_json_response(raw_content: str) -> str:
    """
    清理 LLM 返回的 JSON，去除 markdown 代码块标记
    """
    content = raw_content.strip()
    
    # 去除 markdown 代码块标记
    if content.startswith("```json"):
        content = content[7:]  # 去除 ```json
    elif content.startswith("```"):
        content = content[3:]  # 去除 ```
    
    if content.endswith("```"):
        content = content[:-3]  # 去除结尾的 ```
    
    return content.strip()


def call_llm(messages: List[Dict], output_structure: BaseModel):
    # Initialize OpenAI client
    client = OpenAI(base_url = OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    
    # 在最后一条用户消息中添加 JSON 格式要求
    if messages[-1]["role"] == "user":
        messages[-1]["content"] += """\n\nIMPORTANT: You MUST respond with ONLY a valid JSON object matching this schema, with NO additional text, explanations, or markdown formatting:
{
  "alphas": [
    {
      "alpha_expression": "string",
      "economic_rationale": "string", 
      "data_fields_used": ["string"],
      "operators_used": ["string"]
    }
  ]
}"""
    
    # Call OpenAI API with basic JSON mode
    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=messages,
        response_format={"type": "json_object"},  # 基础 JSON 模式
        temperature=0.8,
        max_tokens=4000
    )
    
    # Extract and parse JSON response
    raw_content = completion.choices[0].message.content
    
    # 清理 JSON（去除 markdown 代码块等）
    cleaned_content = clean_json_response(raw_content)
    
    # 手动解析 JSON 并验证结构
    try:
        llm_structured_output = output_structure.model_validate_json(cleaned_content)
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {raw_content[:500]}...")  # 打印前500字符用于调试
        print(f"Cleaned response: {cleaned_content[:500]}...")  # 打印清理后的内容
        raise
    
    return llm_structured_output

def generate_alpha_expressions(hypothesis: str, operators_reference: str, dataset_reference: str, num_alphas: int = 10) -> AlphaExpressions:

    
    # Construct prompts for alpha generation

    system_prompt = "You are an expert quantitative researcher with deep knowledge of WorldQuant Brain alpha expressions, financial markets, and quantitative trading strategies."
    
    user_prompt = f"""You are an expert quantitative researcher specializing in creating alpha expressions for WorldQuant Brain platform.

Given the following investment hypothesis or idea:
"{hypothesis}"

Generate {num_alphas} diverse alpha expressions that capture different aspects of this hypothesis.

Here are the dataset details and available data fields:

{dataset_reference}

Here is the operators reference:

{operators_reference}

IMPORTANT REQUIREMENTS:

1. **WorldQuant Brain Syntax**: Use proper WorldQuant Brain syntax and strictly use provided operators references.
   - **CRITICAL**: Ensure all operators requiring lookback (like ts_mean, ts_zscore, ts_delta, ts_std_dev) have a valid positive INTEGER lookback parameter (e.g., 5, 20, 60, 120, 252, 500).
   - Example: `ts_mean(close, 20)` NOT `ts_mean(close)` or `ts_mean(close, 0.5)`
   - Do NOT use floats for lookback periods (e.g., use 20 not 20.0).
   - Do NOT use 0 for lookback periods.
   - **CRITICAL**: Do NOT use ts_regression(y, x, d, 0, 0), should use ts_regression(y, x, d, lag = 0, rettype = 0)。
   - ts_rank(x, d, constant = 0) Do NOT use ts_rank(x, d, 0).
   - ts_quantile(x,d, driver="gaussian" ) Do NOT use ts_quantile(x,d, "gaussian").
   
   **ADDITIONAL SYNTAX RULES (CRITICAL - violations cause ERRORS):**
   - `scale(x)` takes ONLY 1 input - use `scale(expression)` NOT `scale(expression, 2)`
   - `quantile(x)` for cross-sectional quantile takes ONLY 1 input
   - `hump(x, hump=threshold)` - the threshold must be a constant number like 0.05
   - `power(x, n)` NOT `pow(x, n)` - use the correct function name
   - `inverse(x)` takes ONLY 1 input - this means 1/x
   - `winsorize(x, std=limit)` - limit is a constant like 3,4,5
   - `ts_returns(x, d, mode=1)` - mode MUST be keyword arg, NOT positional
   - `group_neutralize(x, sector)` - group name is NOT quoted, use bare word: sector, industry, subindustry
   - `ts_rank(x, d, constant=0)` - constant MUST be keyword arg
   - Do NOT nest operators incorrectly - each operator has a specific signature
   
   **PREFER SIMPLE, RELIABLE OPERATORS:**
   - Prefer: rank, ts_delta, ts_mean, ts_std_dev, ts_zscore, multiply, divide, add, subtract
   - Avoid complex operators like: quantile, hump, ts_kurtosis, ts_regression unless you know exact syntax

2. **Expression Diversity**: Each alpha should:
   - Use different combinations of operators and data fields
   - Capture different aspects of the hypothesis (momentum, value, quality, volatility, etc.)
   - Vary in complexity (some simple, some more sophisticated)
   - Consider different time horizons (short-term vs long-term)

3. **Economic Rationale**: Provide clear, concise economic reasoning that:
   - Explains WHY the alpha should work
   - Links back to the original hypothesis
   - Describes the market inefficiency or behavioral bias being exploited
   - Is specific to the alpha expression (not generic)

4. **Data Fields**: List ALL data fields actually used in the expression
   - Be specific (e.g., "close", "volume", "market_cap")
   - Include derived fields if relevant

5. **Operators**: List ALL operators used in the expression
   - Include both time-series and cross-sectional operators
   - Be comprehensive and accurate

EXAMPLE FORMAT (for reference):
- Alpha Expression: rank(ts_delta(close, 5) / ts_std_dev(close, 20))
- Economic Rationale: Captures short-term momentum normalized by recent volatility, identifying stocks with strong recent price moves relative to their volatility
- Data Fields: ["close"]
- Operators: ["rank", "ts_delta", "ts_std_dev"]

Now generate {num_alphas} high-quality alpha expressions based on the hypothesis provided."""

    # Call LLM
    messages=[
            {
                "role": "system",
                "content": system_prompt          },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    
    # Extract structured output
    alpha_expressions = call_llm(messages, AlphaExpressions)
    
    # 清理生成的 alpha 表达式：去除换行符，确保是单行字符串
    for alpha in alpha_expressions.alphas:
        # 将换行符替换为空格，并去除首尾空白
        clean_expr = alpha.alpha_expression.replace('\n', ' ').strip()
        # 将多个空格合并为一个空格
        clean_expr = ' '.join(clean_expr.split())
        alpha.alpha_expression = clean_expr

    return alpha_expressions

def print_alpha_expressions(alpha_expressions: AlphaExpressions):
    """
    Pretty print the generated alpha expressions.
    
    Args:
        alpha_expressions: AlphaExpressions object to print
    """
    print(f"\n{'='*80}")
    print(f"已生成 {len(alpha_expressions.alphas)} 条 Alpha 表达式")
    print(f"{'='*80}\n")
    
    for i, alpha in enumerate(alpha_expressions.alphas, 1):
        print(f"Alpha #{i}")
        print(f"{'-'*80}")
        print(f"表达式: {alpha.alpha_expression}")
        print(f"\n经济合理解释:\n{alpha.economic_rationale}")
        print(f"\n使用的字段: {', '.join(alpha.data_fields_used)}")
        print(f"使用的算子: {', '.join(alpha.operators_used)}")
        print(f"\n{'='*80}\n")


def export_to_dict(alpha_expressions: AlphaExpressions) -> List[dict]:
    return [alpha.model_dump() for alpha in alpha_expressions.alphas]


def generate_alpha_with_templates(
    hypothesis: str,
    template_reference: str,
    field_whitelist: str,
    operators_reference: str,
    num_alphas: int = 10
) -> List[Dict]:
    """
    Template-constrained alpha generation (优化.md Section 3.3, 5 改法A).
    LLM outputs JSON specs, which are then rendered to valid expressions.
    """
    from openai import OpenAI
    
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    
    system_prompt = """You are an expert quantitative researcher. You generate alpha expressions by selecting from predefined templates and filling in parameters. You MUST output valid JSON only."""
    
    user_prompt = f"""Given this investment hypothesis:
"{hypothesis}"

Generate {num_alphas} alpha expressions using the TEMPLATES below.

**AVAILABLE TEMPLATES (you MUST use these):**
{template_reference}

**HIGH-POTENTIAL FIELDS (prioritize these):**
{field_whitelist}

**OPERATORS REFERENCE:**
{operators_reference}

**OUTPUT FORMAT:**
Return a JSON object with this exact structure:
{{
  "alphas": [
    {{
      "template_id": "momentum_rank",
      "field": "mdl110_growth",
      "window": 20,
      "economic_rationale": "Captures growth momentum..."
    }}
  ]
}}

Rules:
1. template_id MUST match one of the available template IDs
2. Use ONLY the high-potential fields listed above when possible
3. window/window1/window2/short_window/long_window should be integers from [5, 10, 20, 60, 120, 252]
4. Each alpha should be diverse - different template + field combinations"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=4000
        )
        
        raw_content = completion.choices[0].message.content
        cleaned = clean_json_response(raw_content)
        specs = json.loads(cleaned)
        
        # Render each spec to an expression
        alpha_dicts = []
        for spec in specs.get('alphas', []):
            try:
                expression = render_expression(spec)
                alpha_dicts.append({
                    'alpha_expression': expression,
                    'economic_rationale': spec.get('economic_rationale', ''),
                    'template_id': spec.get('template_id'),
                    'spec': spec
                })
                print(f"  ✓ 渲染成功: {expression}...")
            except Exception as e:
                print(f"  ✗ 渲染失败: {e}")
        
        return alpha_dicts
        
    except Exception as e:
        print(f"模板生成过程出错: {e}")
        return []


def optimize_weak_alphas(
    alpha_dicts: List[Dict],
    brain_session,
    region: str,
    universe: str,
    delay: int = 1,
    neutralization: str = "INDUSTRY",
    decay: int = 4,
    truncation: float = 0.02,
    test_period: str = "P2Y",
    max_variants_per_alpha: int = 5
) -> List[Dict]:
    """
    优化链：针对弱 Alpha 生成局部改写（优化.md 第 3.4 节）。
    """
    print(f"\n{'='*80}")
    print(f"优化链：正在处理弱 Alpha")
    print(f"{'='*80}\n")
    
    optimized_alphas = []
    
    for alpha_dict in alpha_dicts:
        # 跳过模拟失败的
        if alpha_dict.get('simulation_status') != 'success':
            optimized_alphas.append(alpha_dict)
            continue
        
        sim_result = alpha_dict.get('simulation', {})
        # 有信号的alpha才进行优化
        should_opt, reason = should_optimize(sim_result)
        
        if not should_opt:
            print(f"  ✓ {alpha_dict['alpha_expression']}... - 指标足够好，跳过优化")
            optimized_alphas.append(alpha_dict)
            continue
        
        print(f"  → 正在优化: {alpha_dict['alpha_expression']}... (原因: {reason})")
        
        # 生成局部改写
        variants = generate_local_rewrites(
            alpha_dict['alpha_expression'],
            sim_result,
            feedback=reason,
            max_variants=max_variants_per_alpha
        )
        
        if not variants:
            optimized_alphas.append(alpha_dict)
            continue
        
        # 准备模拟配置
        variant_configs = []
        for var in variants:
            config = ace.generate_alpha(
                regular=var['expression'],
                alpha_type="REGULAR",
                region=region,
                universe=universe,
                delay=delay,
                neutralization=neutralization,
                decay=decay,
                truncation=truncation,
                pasteurization="ON",
                test_period=test_period,
                unit_handling="VERIFY",
                nan_handling="ON",
                max_trade="OFF",
                visualization=False,
            )
            variant_configs.append({'config': config, 'variant': var})
        
        # 模拟变体（批量模拟以提高效率）
        try:
            configs_only = [vc['config'] for vc in variant_configs]
            results = ace.simulate_alpha_list_multi(
                brain_session,
                configs_only,
                limit_of_concurrent_simulations=4,
                limit_of_multi_simulations=4
            )
            
            # 寻找最佳变体
            best_score = calculate_alpha_score(sim_result)  # 原始分数
            best_alpha = alpha_dict
            
            for res in results:
                if res.get('alpha_id'):
                    score = calculate_alpha_score(res)
                    if score > best_score:
                        best_score = score
                        best_alpha = {
                            'alpha_expression': res.get('simulate_data', {}).get('regular', ''),
                            'economic_rationale': alpha_dict.get('economic_rationale', '') + ' [已优化]',
                            'simulation': res,
                            'simulation_status': 'success',
                            'original_expression': alpha_dict['alpha_expression']
                        }
            
            if best_alpha != alpha_dict:
                print(f"    ✓ 分数提升: {calculate_alpha_score(sim_result):.3f} → {best_score:.3f}")
            else:
                print(f"    - 未找到更好的变体")
            
            optimized_alphas.append(best_alpha)
            
        except Exception as e:
            print(f"    ✗ 优化失败: {e}")
            optimized_alphas.append(alpha_dict)
    
    return optimized_alphas

class SimResultsEncoder(json.JSONEncoder):
    """Custom encoder for simulation results containing DataFrames"""
    def default(self, obj):
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        return super().default(obj)

def export_to_json(alpha_dicts: List[Dict]):
    with open('alphas.json', 'w') as json_file:
        json.dump(alpha_dicts, json_file, indent=4, cls=SimResultsEncoder)

def validate_api_credentials():
    """验证 API 凭证是否已设置"""
    if not OPENAI_API_KEY or not OPENAI_BASE_URL:
        print("\n" + "="*80)
        print("错误: API 凭证未配置！")
        print("="*80)
        print("\n请设置以下环境变量:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("  export OPENAI_BASE_URL='https://api.deepseek.com/v1'")
        print("\n或者在 test.py 的第 88-89 行直接硬编码您的凭证（不推荐用于生产环境）")
        print("="*80 + "\n")
        return False
    return True


def simulate_alphas_batch(alpha_dicts: List[Dict], brain_session, 
                          region="USA", universe="TOP3000", 
                          delay=1, neutralization="SECTOR",
                          decay=4, truncation=0.02,
                          test_period="P2Y") -> List[Dict]:
    """
    批量模拟 alpha 表达式 (使用 ace.simulate_alpha_list_multi)
    
    Args:
        alpha_dicts: Alpha 字典列表
        brain_session: BRAIN session
        其他参数: 模拟设置
    
    Returns:
        更新后的 alpha_dicts，包含模拟结果
    """
    print(f"\n{'='*80}")
    print(f"正在开始使用 simulate_alpha_list_multi 批量模拟 {len(alpha_dicts)} 条 Alpha")
    print(f"{'='*80}\n")
    
    # 1. 准备配置列表
    alpha_configs = []
    for alpha_dict in alpha_dicts:
        # 使用 generate_alpha 生成配置字典 (不进行模拟)
        config = ace.generate_alpha(
            regular=alpha_dict['alpha_expression'],
            alpha_type="REGULAR",
            region=region,
            universe=universe,
            delay=delay,
            neutralization=neutralization,
            decay=decay,
            truncation=truncation,
            pasteurization="ON",
            test_period=test_period,
            unit_handling="VERIFY",
            nan_handling="ON",
            max_trade="OFF",
            visualization=False,
        )
        alpha_configs.append(config)

    # 2. 调用批量模拟
    try:
        # 降低并发数以避免 API 限制
        # limit_of_multi_simulations: 每个批次包含的 alpha 数量
        # limit_of_concurrent_simulations: 同时进行的批次数量
        print("正在发送模拟请求...")
        simulation_results = ace.simulate_alpha_list_multi(
            brain_session, 
            alpha_configs,
            limit_of_concurrent_simulations=5,  # 降低并发
            limit_of_multi_simulations=5        # 降低每个批次的大小
        )
    except Exception as e:
        print(f"simulate_alpha_list_multi 过程出错: {e}")
        # Mark all as failed if the batch call itself crashes
        for alpha_dict in alpha_dicts:
            alpha_dict['simulation_status'] = 'failed'
            alpha_dict['simulation_error'] = str(e)
        return alpha_dicts

    # 3. 将结果映射回 alpha_dicts
    successful = 0
    failed = 0
    
    # 创建一个映射: alpha_expression -> result
    expr_to_result = {}
    for res in simulation_results:
        sim_data = res.get('simulate_data', {})
        # ace_lib 可能会返回处理过的 regular 字段
        # 我们尝试匹配 regular 字段
        expr = sim_data.get('regular')
        if expr:
            # 清理表达式以便匹配（去除可能的换行符差异）
            clean_expr_key = expr.replace('\n', ' ').strip()
            expr_to_result[clean_expr_key] = res

    for alpha_dict in alpha_dicts:
        # 同样清理字典中的表达式以便匹配
        original_expr = alpha_dict['alpha_expression']
        clean_expr = original_expr.replace('\n', ' ').strip()
        
        if clean_expr in expr_to_result:
            res = expr_to_result[clean_expr]
            
            # 检查是否有有效 alpha_id
            if res.get('alpha_id'):
                alpha_dict['simulation'] = res
                alpha_dict['simulation_status'] = 'success'
                print(f"  ✓ 成功！Alpha ID: {res['alpha_id']}")
                successful += 1
            else:
                alpha_dict['simulation_status'] = 'failed'
                alpha_dict['simulation_error'] = "未返回 alpha_id"
                print(f"  ✗ 失败: {original_expr}... 未返回 alpha_id")
                failed += 1
        else:
            alpha_dict['simulation_status'] = 'failed'
            alpha_dict['simulation_error'] = "未找到匹配该表达式的模拟结果"
            print(f"  ✗ 失败: 未找到 {original_expr}... 的模拟结果")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"批量模拟完成: {successful} 条成功, {failed} 条失败")
    print(f"{'='*80}\n")
    
    return alpha_dicts


def add_llm_tags_and_descriptions(alpha_dicts: List[Dict], brain_session, 
                                   llm_tag: str = "deepseek-v3.2") -> List[Dict]:
    """
    为所有 alpha 添加 LLM 标签和描述
    
    Args:
        alpha_dicts: Alpha 字典列表
        brain_session: BRAIN session
        llm_tag: LLM 标签（例如：deepseek-v3.2, gpt-4o 等）
    
    Returns:
        更新后的 alpha_dicts
    """
    print(f"\n{'='*80}")
    print(f"正在添加 LLM 标签和描述")
    print(f"{'='*80}\n")
    
    for i, alpha_dict in enumerate(alpha_dicts, 1):
        if alpha_dict.get('simulation_status') != 'success':
            print(f"[{i}/{len(alpha_dicts)}] 跳过模拟失败的 Alpha")
            continue
            
        try:
            alpha_id = alpha_dict['simulation'].get('alpha_id')
            if not alpha_id:
                print(f"[{i}/{len(alpha_dicts)}] 未找到 alpha_id，跳过")
                continue
            
            # 添加标签和描述
            description = alpha_dict['economic_rationale']
            
            print(f"[{i}/{len(alpha_dicts)}] 正在更新 Alpha {alpha_id}...")
            
            # 使用 ace.set_alpha_properties 设置标签和描述
            ace.set_alpha_properties(
                brain_session,
                alpha_id=alpha_id,
                tags=[llm_tag],
                combo_desc=description[:150]  # 限制描述长度
            )
            
            alpha_dict['llm_tag'] = llm_tag
            alpha_dict['description_added'] = True
            print(f"  ✓ 已更新标签 '{llm_tag}' 和描述")
            
        except Exception as e:
            alpha_dict['description_added'] = False
            print(f"  ✗ 更新失败: {str(e)}")
    
    print(f"\n{'='*80}")
    print(f"LLM 标签和描述添加完成")
    print(f"{'='*80}\n")
    
    return alpha_dicts

def get_default_universe(region):
    if region == "USA":
        return "TOP3000"
    elif region == "EUR":
        return "TOP2500"
    elif region == "ASI":
        return "MINVOL1M"
    elif region == "GLB":
        return "TOP3000"
    elif region == "IND":
        return "TOP500"
    elif region == "KOR":
        return "TOP600"
    else:
        return "TOP3000"
def perform_grid_search(alpha_dicts: List[Dict], brain_session, 
                        region: str, universe: str, test_period: str,
                        search_grid: Dict[str, List]) -> Dict:
    """
    Performs a grid search over simulation settings to find the best configuration.
    """
    import copy
    import itertools
    import numpy as np
    
    # Generate all combinations
    keys = list(search_grid.keys())
    values = list(search_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"\n{'='*80}")
    print(f"正在开始网格搜索：共 {len(combinations)} 种组合")
    print(f"网格配置：{search_grid}")
    print(f"{'='*80}\n")
    
    best_score = -float('inf')
    best_config = None
    results_history = []
    
    for i, combo in enumerate(combinations, 1):
        # Create config dict for this iteration
        current_config = dict(zip(keys, combo))
        print(f"\n[网格搜索 {i}/{len(combinations)}] 正在测试配置：{current_config}")
        
        # Deep copy alphas to avoid polluting state between runs
        # (Though simulate_alphas_batch modifies in place, we want fresh start for each config if we want to isolate errors, 
        # but mainly we need to reset simulation results)
        # Actually, simulate_alphas_batch adds 'simulation' key. We should use a fresh copy.
        current_alphas = copy.deepcopy(alpha_dicts)
        
        # Run simulation
        simulated_alphas = simulate_alphas_batch(
            current_alphas,
            brain_session,
            region=region,
            universe=universe,
            delay=1, # Fixed for now or could be in grid
            test_period=test_period,
            neutralization=current_config.get('neutralization', 'INDUSTRY'),
            decay=current_config.get('decay', 4),
            truncation=current_config.get('truncation', 0.02)
        )
        
        # Calculate Score using multi-objective scoring
        from alpha_scoring import calculate_alpha_score
        
        scores = []
        for alpha in simulated_alphas:
            if alpha.get('simulation_status') == 'success':
                sim_res = alpha.get('simulation', {})
                score = calculate_alpha_score(sim_res)
                scores.append(score)
                alpha['composite_score'] = score  # Store for later use
        
        if scores:
            avg_score = np.mean(scores)
            count = len(scores)
        else:
            avg_score = -999.0
            count = 0
            
        print(f"  -> 结果：平均分数：{avg_score:.4f} (基于 {count} 条成功的 Alpha)")
        
        # Record history
        record = {
            'config': current_config,
            'avg_score': avg_score,
            'successful_count': count
        }
        results_history.append(record)
        
        # Update best
        if avg_score > best_score:
            best_score = avg_score
            best_config = current_config
            print(f"  -> 发现新的最佳配置！")

    print(f"\n{'='*80}")
    print(f"网格搜索完成")
    print(f"最高分数：{best_score:.4f}")
    print(f"最佳配置：{best_config}")
    print(f"{'='*80}\n")
    
    # Save best config
    if best_config:
        output_file = f"best_settings_{region}_{dataset_id}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'best_config': best_config,
                'best_score': best_score,
                'history': results_history
            }, f, indent=4)
        print(f"已将最佳设置保存至 {output_file}")
            
    return best_config

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    dataset_id="model110"
    region="USA"
    test_period="P2Y"
    universe=get_default_universe(region)
    
    # Configuration flags
    USE_FIELD_SCREENING = True      # Enable field screening (优化.md 3.2)
    USE_TEMPLATE_GENERATION = True  # Enable template-based generation (优化.md 3.3)
    USE_OPTIMIZATION_CHAIN = True   # Enable optimization chain (优化.md 3.4)
    
    # Step 1: 验证 API 凭证
    if not validate_api_credentials():
        print("由于缺少 API 凭证，正在退出。")
        exit(1)
    
    # Step 2: 准备参考资料
    print("正在加载算子和数据集参考资料...")
    operators_reference = get_operators_reference(s)
    dataset_reference = get_dataset_reference(s, dataset_id=dataset_id)
    
    # Step 3: 定义研究假设
    hypothesis = "营收增长持续上升且波动性较低的股票表现优于市场"
    print(f"\n正在根据以下研究假设生成 Alpha 表达式：\n'{hypothesis}'\n")
    
    # ========================================================================
    # NEW: Phase 1 - 字段筛选 (优化.md Section 3.2)
    # ========================================================================
    field_whitelist = ""
    if USE_FIELD_SCREENING:
        print(f"\n{'='*80}")
        print("阶段 1: 字段筛选 - 寻找高潜力字段")
        print(f"{'='*80}\n")
        
        try:
            # Get available fields from dataset
            from ace_lib import get_datafields
            fields_df = get_datafields(s, dataset_id=dataset_id, region=region, universe=universe)
            
            # Use simple screening: rank fields by alphaCount (proxy for information content)
            if 'alphaCount' in fields_df.columns:
                top_fields_df = fields_df.nlargest(10, 'alphaCount')
                field_list = top_fields_df['id'].tolist()
                field_whitelist = "推荐的高潜力数据字段（优先使用）：\n"
                for f in field_list:
                    field_whitelist += f"  - {f}\n"
                print(f"找到 {len(field_list)} 个高潜力字段")
            else:
                # If no alphaCount, just take first 10 fields
                field_list = fields_df['id'].head(10).tolist()
                field_whitelist = "可用数据字段：\n" + "\n".join([f"  - {f}" for f in field_list])
                print(f"使用前 {len(field_list)} 个字段（无排名信息可用）")
        except Exception as e:
            print(f"字段筛选失败: {e}，将使用所有字段")
            field_whitelist = ""
    
    # ========================================================================
    # NEW: Phase 2 - 基于模板的 Alpha 生成 (优化.md Section 3.3)
    # ========================================================================
    if USE_TEMPLATE_GENERATION and field_whitelist:
        print(f"\n{'='*80}")
        print("阶段 2: 基于模板的 Alpha 生成")
        print(f"{'='*80}\n")
        
        template_reference = get_template_prompt_reference()
        alpha_dicts_original = generate_alpha_with_templates(
            hypothesis=hypothesis,
            template_reference=template_reference,
            field_whitelist=field_whitelist,
            operators_reference=operators_reference,
            num_alphas=10
        )
        print(f"\n✓ 已生成 {len(alpha_dicts_original)} 条基于模板的 Alpha 表达式")
    else:
        # Fallback to original LLM generation
        print(f"\n{'='*80}")
        print("阶段 2: LLM 直接生成 (备选方案)")
        print(f"{'='*80}\n")
        
        result = generate_alpha_expressions(
            hypothesis, 
            operators_reference, 
            dataset_reference, 
            num_alphas=10
        )
        print_alpha_expressions(result)
        alpha_dicts_original = export_to_dict(result)
        print(f"\n✓ 已生成 {len(alpha_dicts_original)} 条 Alpha 表达式")
    
    # ========================================================================
    # Phase 3: 初始模拟 (优化.md Section 3.4)
    # ========================================================================
    print(f"\n{'='*80}")
    print("阶段 3: 初始模拟")
    print(f"{'='*80}\n")
    
    initial_alphas = simulate_alphas_batch(
        alpha_dicts_original,
        s,
        region=region,
        universe=universe,
        delay=1,
        neutralization="INDUSTRY",
        decay=4,
        truncation=0.02,
        test_period=test_period
    )
    
    # ========================================================================
    # NEW: Phase 4 - 优化链 (优化.md Section 3.4)
    # ========================================================================
    if USE_OPTIMIZATION_CHAIN:
        print(f"\n{'='*80}")
        print("阶段 4: 优化链 - 改进弱 Alpha")
        print(f"{'='*80}\n")
        
        optimized_alphas = optimize_weak_alphas(
            initial_alphas,
            s,
            region=region,
            universe=universe,
            delay=1,
            neutralization="INDUSTRY",
            decay=4,
            truncation=0.02,
            test_period=test_period,
            max_variants_per_alpha=5
        )
    else:
        optimized_alphas = initial_alphas
    
    # ========================================================================
    # Phase 5: 环境参数网格搜索 (优化.md Section 3.5)
    # ========================================================================
    print(f"\n{'='*80}")
    print("阶段 5: 环境参数网格搜索")
    print(f"{'='*80}\n")
    
    search_grid = {
        'neutralization': ['INDUSTRY', 'SECTOR', 'SUBINDUSTRY', 'MARKET', 'CROWDING', 'REVERSE_AND_MOMENTUM', 'FAST'],
        'decay': [1, 4, 8, 15],
        'truncation': [0.01, 0.05]
    }
    
    best_config = perform_grid_search(
        optimized_alphas,
        s,
        region=region,
        universe=universe,
        test_period=test_period,
        search_grid=search_grid
    )
    
    # Final pass with best settings
    if best_config:
        print("\n正在使用最佳设置执行最终模拟...")
        final_alphas = simulate_alphas_batch(
            optimized_alphas,
            s,
            region=region,
            universe=universe,
            delay=1,
            neutralization=best_config['neutralization'],
            decay=best_config['decay'],
            truncation=best_config['truncation'],
            test_period=test_period
        )
        
        # Add LLM tags and descriptions
        final_alphas = add_llm_tags_and_descriptions(
            final_alphas, 
            s,
            llm_tag=os.getenv("OPENAI_MODEL")
        )
        
        # Export to JSON
        export_to_json(final_alphas)
        print(f"\n✓ 成功以最佳设置导出结果")
    else:
        final_alphas = optimized_alphas
        export_to_json(final_alphas)
    
    # Print summary
    successful_count = sum(1 for a in final_alphas if a.get('simulation_status') == 'success')
    print(f"\n{'='*80}")
    print("总结报告")
    print(f"{'='*80}")
    print(f"执行阶段：")
    print(f"  1. 字段筛选: {'✓' if USE_FIELD_SCREENING else '✗'}")
    print(f"  2. 模板生成: {'✓' if USE_TEMPLATE_GENERATION else '✗'}")
    print(f"  3. 初始模拟: ✓")
    print(f"  4. 优化链: {'✓' if USE_OPTIMIZATION_CHAIN else '✗'}")
    print(f"  5. 网格搜索: ✓")
    print(f"")
    print(f"总计生成的 Alpha 数量: {len(final_alphas)}")
    print(f"成功模拟的数量: {successful_count}")
    print(f"模拟失败的数量: {len(final_alphas) - successful_count}")
    print(f"输出文件: alphas.json")
    print(f"{'='*80}\n")