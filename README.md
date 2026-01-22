# AIAC 2.0 - LLM Alpha Generator

使用 LLM 从研究假设自动生成 WorldQuant Brain Alpha 表达式的完整工具。

## 📋 功能特性

✅ **OpenAI 兼容 API 支持** - 适配所有 OpenAI 兼容的 LLM API  
✅ **自动 JSON 清理** - 智能处理 markdown 包裹的 JSON 响应  
✅ **批量模拟** - 使用 `ace.simulate_alpha_list_multi` 并发模拟，效率更高  
✅ **自动标签** - 为每个 alpha 添加 LLM 标签和经济学描述  
✅ **错误处理** - 完善的错误处理和进度显示  

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install pandas tabulate openai pydantic
```

### 2. 配置 API 凭证

#### 方法 A：环境变量（推荐）

```bash
export OPENAI_API_KEY="sk-xxxxxx"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
```

#### 方法 B：代码硬编码（仅用于测试）

编辑 `test.py` 第 88-89 行：

```python
OPENAI_API_KEY = "sk-your-actual-key-here"
OPENAI_BASE_URL = "https://api.deepseek.com/v1"
```

### 3. 运行脚本

```bash
python test.py
```

---

## 📖 代码结构说明

### 核心函数

| 函数名 | 功能 |
|--------|------|
| `clean_json_response()` | 清理 LLM 返回的 JSON（去除 markdown 标记） |
| `call_llm()` | 调用 LLM API，支持所有 OpenAI 兼容接口 |
| `get_operators_reference()` | 获取 BRAIN 可用操作符列表 |
| `get_dataset_reference()` | 获取数据集和字段信息 |
| `generate_alpha_expressions()` | 基于研究假设生成 alpha 表达式 |
| `simulate_alphas_batch()` | 批量模拟 alpha 表达式 (使用 `ace.simulate_alpha_list_multi`) |
| `add_llm_tags_and_descriptions()` | 添加 LLM 标签和描述 |

### 工作流程

```
1. 验证 API 凭证
   ↓
2. 加载操作符和数据集参考
   ↓
3. 定义研究假设
   ↓
4. 调用 LLM 生成 alpha 表达式
   ↓
5. 批量模拟表达式
   ↓
6. 添加 LLM 标签和描述
   ↓
7. 导出到 alphas.json
```

---

## 🔧 关键修改点

### 从 OpenAI Structured Outputs 到兼容模式

**原始代码问题：**
```python
# ❌ 不兼容大部分 API
completion = client.chat.completions.parse(
    response_format=output_structure
)
```

**修复后代码：**
```python
# ✅ 兼容所有 OpenAI 兼容 API
completion = client.chat.completions.create(
    response_format={"type": "json_object"}
)
raw_content = completion.choices[0].message.content
cleaned_content = clean_json_response(raw_content)
llm_structured_output = output_structure.model_validate_json(cleaned_content)
```

### JSON 清理功能

处理模型返回的各种格式：

```python
# 输入：
"""
```json
{
  "alphas": [...]
}
```
"""

# 输出：纯净的 JSON 字符串
{"alphas": [...]}
```

---

## 📝 自定义配置

### 修改研究假设

编辑 `test.py` 第 409 行：

```python
hypothesis = "Your research idea here"
```

### 修改模拟参数

编辑 `test.py` 第 426-434 行：

```python
alpha_dicts = simulate_alphas_batch(
    alpha_dicts, 
    s,
    region="USA",           # 市场区域
    universe="TOP1000",     # 股票池
    delay=1,                # 延迟
    neutralization="SECTOR", # 中性化方式
    decay=4,                # 衰减
    truncation=0.02,        # 截断
    test_period="P2Y"       # 测试周期
)
```

### 修改 LLM 标签

编辑 `test.py` 第 442 行，根据你使用的 LLM 修改标签：

```python
llm_tag="DEEPSEEK"  # 可选：GPT4, CLAUDE, GEMINI 等
```

---

## 🐛 常见问题

### 1. API Key 错误

**错误：** `OpenAIError: The api_key client option must be set`

**解决：**
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="your-base-url"
```

### 2. JSON 解析错误

**错误：** `ValidationError: Invalid JSON: expected value at line 1`

**原因：** 模型返回了 markdown 包裹的 JSON

**解决：** 代码已自动处理，如果仍有问题，检查 `clean_json_response()` 函数

### 3. 模拟失败

**原因：** 
- Alpha 表达式语法错误
- 使用了不存在的字段
- 模拟参数不匹配

**解决：** 查看错误日志，调整 prompt 或模拟参数

---

## 📊 输出格式

最终输出 `alphas.json`，包含以下字段：

```json
[
  {
    "alpha_expression": "rank(multiply(ts_delta(mdl110_growth, 20), ...))",
    "economic_rationale": "Captures short-term momentum...",
    "data_fields_used": ["mdl110_growth"],
    "operators_used": ["rank", "multiply", "ts_delta"],
    "simulation": {
      "alpha_id": "12345678",
      ...
    },
    "simulation_status": "success",
    "llm_tag": "DEEPSEEK",
    "description_added": true
  }
]
```

---

## 🎯 支持的 LLM 供应商

只要支持 OpenAI 兼容协议，都可以使用：

| 供应商 | Base URL 示例 |
|--------|---------------|
| DeepSeek | `https://api.deepseek.com/v1` |
| 硅基流动 | `https://api.siliconflow.cn/v1` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 月之暗面 | `https://api.moonshot.cn/v1` |
| OpenAI | `https://api.openai.com/v1` |

---

## 📞 支持

如有问题，请参考：
- ACE Library 文档
- WorldQuant BRAIN FAQ
- Competition Guidelines

---

**Good luck with AIAC 2.0! 🚀**
