# 王玺尊个人模块 README

> 人工智能实训 B 方向个人模块说明。负责范围为 **B4 Agent LLM 决策模块**、**B5 记忆文档存储与查找模块**，并与团队成员协作完成前端模块页面、系统联调和工程整理。

---

## 1. 模块概述

### 1.1 模块名称

- `B4：Agent LLM 决策模块`
- `B5：记忆文档存储与查找模块`
- `B4/B5 前端观察与演示页面（协作）`

### 1.2 模块说明

#### B4：模型通信与输出协议层

B4 位于 B1 Agent Runtime 与模型服务之间。模块接收 B1 提供的 messages、B3 生成的 tools schema 和模型配置，调用 `local`、`fastapi` 或 `qwen_api` 模型源，再将模型原始输出解析为标准 AIMessage 或阶段 JSON。

B4 只处理模型通信和协议转换：

- 不执行 B2 Skill；
- 不代替 B1 控制 Agent Loop；
- 不把业务关键词硬编码为工具选择；
- 不保存或检索 B5 长期记忆。

该边界保证模型来源和输出协议可以独立演进，而不会污染工具执行与 Agent 编排。

#### B5：事实持久化与分层记忆层

B5 同时保留两条路径：

1. **课程基础兼容路径**：使用 `memory_index.json + Markdown` 完成指定记忆读取、全局记忆加载、长度限制和记忆保存；
2. **浏览器系统主路径**：使用 SQLite 保存会话、原始消息和工具步骤，并生成轮级摘要、记忆块、任务记忆、向量缓存和召回日志。

B5 的设计原则是“原始消息和工具步骤是事实来源，摘要和记忆块只用于定位与筛选”。召回结果保留 source message/tool step id，避免压缩文本替代原始事实。

#### 前端协作

前端由团队协作完成。本人重点参与 B2–B5 模块页面、B4 调用观察与协议演示、B5 记忆快照与召回展示、模块页公共展示逻辑整理；主对话、B1 页面和跨页面状态由团队共同联调。前端只展示和操作真实接口，不在浏览器中重新实现 B4 解析或 B5 召回算法。

### 1.3 完成情况概览

| 类型 | 完成情况 |
|---|---|
| 基础要求 | B4 已实现模型配置读取、tools schema 注入、模型调用、raw output 保存、AIMessage 解析和日志；B5 已实现 legacy memory 的查找、截断、保存、索引和日志，并接入 SQLite 会话持久化。 |
| 进阶要求 | B4 已支持多 tool_calls、多 ToolMessage、流式输出、三类模型源和有限协议容错；B5 已实现轮摘要、记忆块、任务记忆、字段/关键词评分、向量召回、LLM rerank、来源回查和失败降级。 |
| 可独立演示 | B4：`code/b4_local_agent_llm.py`；B5 legacy：`code/b5_memory.py`；浏览器：B4/B5 模块页及对应后端接口。 |
| 团队集成 | B1 调用 B4 取得 AIMessage/阶段 JSON；B1 和后端调用 B5 准备上下文、保存事实并调度后台反思；前端读取真实 B4/B5 产物。 |
| 未完成项 | 自动模型路由、模型协议/多模型/token 对照、任意 Memory 显式冲突合并、错误 Memory 对照实验及固定验收截图尚未完成。 |

### 1.4 个人工作范围与核查依据

| 工作方向 | 重点内容 | 主要文件 | 代表性提交 |
|---|---|---|---|
| B4 | 多模型源适配、prompt JSON、AIMessage 解析、流式输出和协议演示后端 | `code/b4_local_agent_llm.py`、`backend/b4_demo_service.py` | `8b8b3a9`、`ae9c36f` |
| B5 基础与持久化 | legacy memory、SQLite 表结构、会话消息与工具步骤保存 | `code/b5_memory.py`、`code/b5_memory_parts/`、`code/common/conversation_store.py` | `f01eb98`、`242b09d`、`00b26f1` |
| B5 分层记忆 | 轮反思、任务记忆、记忆块、来源证据和受预算上下文 | `reflection.py`、`retrieval.py`、`text_utils.py` | `2c7e263`、`0ba605f` |
| B5 语义检索 | embedding 缓存、向量分数、候选约束 rerank 和降级 | `vector_retrieval.py`、`rerank.py` | `b1a2221`、`0ba605f` |
| 前端与演示 | B4/B5 观察和演示页面、模块展示公共逻辑、B2/B3 页面协作 | `frontend/src/B4*.tsx`、`B5ModuleView.tsx`、`moduleViewUtils.ts` | `b95db9a`、`22a0cbd`、`3a83fe6` |

上述文件经过团队持续联调和共同修改。提交记录用于说明个人重点推进的阶段，不将协作文件表述为个人独占成果。

---

## 2. 环境、模型与数据依赖

### 2.1 运行环境

| 项目 | 要求 |
|---|---|
| Python | 3.10 |
| 必要依赖 | PyYAML、Pydantic、FastAPI、Uvicorn、NumPy；Qwen API 代理使用 `langchain-openai` |
| 前端环境 | Node.js `^20.19.0` 或 `>=22.12.0`，React 19、TypeScript、Vite 8 |
| 是否需要模型 | B4 `prompt_json` 需要；B5 legacy 不需要；B5 反思、向量和 rerank 可使用模型并允许降级 |
| 是否需要 GPU | `qwen_api` / `fastapi` 模式本机不需要；`local` / transformers 模式通常需要 |
| 是否需要外部数据集 | 不需要；使用项目自带消息、文档和 memory 样例 |
| 是否需要联网 | `qwen_api`、远端 `fastapi` 和默认 embedding 需要；本地模型及 legacy memory 可离线 |

### 2.2 模型依赖

模型源由 `configs/model.yaml` 的 `runtime.llm_source` 控制。

| 模型 / 服务 | 来源 | 项目内位置 | 用途 |
|---|---|---|---|
| `qwen-plus` | 阿里云 Model Studio | 无需下载；经 `llm_backend/qwen_api/llm_fastapi_server.py` 代理 | 当前默认 B4 生成，也可用于 B5 反思和 rerank |
| `text-embedding-v4` | 阿里云 embedding API | 无需下载；由 `.env` 和 `configs/memory.yaml` 配置 | B5 候选记忆向量化与相似度召回 |
| `Qwen3.5-4B` | [ModelScope Qwen/Qwen3.5-4B](https://modelscope.cn/models/Qwen/Qwen3.5-4B) | `models/Qwen3.5-4B` | 课程指定的本地模型方案；仓库不包含权重 |
| 远端 FastAPI 模型 | 学校或其他兼容服务 | `configs/model.yaml` 中的 `fastapi.base_url` | 通过统一 `/generate`、`/generate_stream` 接口生成 |

默认 `qwen-plus` 是当前工程配置，不等同于课程指定的本地 `Qwen3.5-4B`。正式验收采用哪一模型，应以实际配置、模型服务状态和运行产物为准。

API 模式的根目录 `.env` 示例：

```dotenv
QWEN_API_KEY=<your-api-key>
QWEN_MODEL=qwen-plus
QWEN_EMBEDDING_MODEL=text-embedding-v4
```

密钥不得写入 README、配置文件或 Git 提交。

### 2.3 数据集或样例数据依赖

项目不训练或微调模型，因此没有训练数据集依赖。

| 数据或文件 | 来源 | 相对路径 | 用途 |
|---|---|---|---|
| B4 初始消息 | 项目自带 | `data/messages/messages_no_tool.json` | 演示直接回答或生成 tool_calls |
| B4 成功工具消息 | 项目自带 | `data/messages/messages_with_tool.json` | 演示基于 ToolMessage 生成最终回答 |
| B4 错误工具消息 | 项目自带 | `data/messages/messages_with_error_tool.json` | 演示工具失败后的回答收束 |
| 工具说明样例 | 项目自带 | `data/messages/tools_schema_basic.json` | B4 tools schema 注入 |
| legacy memory 索引和文档 | 项目自带 | `memory/memory_index.json`、`memory/conversations/conv_000.md` | B5 基础查找演示 |
| legacy 保存输入 | 项目自带 | `data/memory_inputs/memory_save_input.json` | B5 保存消息、轨迹和回答 |
| 会话数据库 | 运行生成 | `memory/conversation_store.sqlite3` | 会话事实、摘要、块、任务、向量缓存和召回日志 |
| B1 阶段提示 | 项目自带 | `prompts/b1_stage_prompts.json` | B4 为 B1 各阶段生成结构化结果 |
| B5 提示 | 项目自带 | `prompts/b5_memory_prompts.json` | B5 反思、任务判断与 rerank |

SQLite 数据库、会话 prompt、检查点和运行输出属于运行时数据，不应当作固定测试集，也不应随意覆盖。

### 2.4 安装步骤

从项目根目录执行：

```bash
conda create -n agent python=3.10 -y
conda activate agent

# 默认 qwen_api / fastapi 环境
pip install -r requirements_fastapi.txt

# local / transformers 模式改用完整依赖，并另行准备兼容的 PyTorch
# pip install -r requirements.txt

cd frontend
npm ci
cd ..
```

本地模型模式需根据运行机器的 CUDA/CPU 环境安装 PyTorch，并确保 `configs/model.yaml` 中的模型路径真实存在。

---

## 3. 文件结构与接口边界

### 3.1 文件结构

```text
agent/
├── code/
│   ├── b4_local_agent_llm.py                 # B4 模型来源、生成、流式和解析
│   ├── b5_memory.py                          # B5 公共接口与 legacy CLI
│   ├── b5_memory_parts/
│   │   ├── legacy.py                         # memory_index + Markdown 兼容路径
│   │   ├── conversation_api.py               # SQLite 会话记忆公共接口
│   │   ├── reflection.py                     # 轮反思、任务记忆和记忆块
│   │   ├── retrieval.py                      # 分层召回与 B1 上下文包
│   │   ├── text_utils.py                     # 评分、来源和字符预算
│   │   ├── vector_retrieval.py               # embedding、缓存和向量分数
│   │   └── rerank.py                         # 候选约束的 LLM 重排
│   └── common/conversation_store.py          # SQLite 表结构与持久化函数
├── configs/
│   ├── model.yaml                            # B4 模型配置
│   └── memory.yaml                           # B5 存储与召回配置
├── prompts/b5_memory_prompts.json            # B5 结构化反思与重排提示
├── llm_backend/
│   ├── qwen_api/llm_fastapi_server.py        # Qwen API 本地代理
│   └── server/llm_fastapi_server.py          # 本地模型 FastAPI 服务
├── backend/
│   ├── b4_demo_service.py                    # B4 观察与协议演示服务
│   ├── main.py                               # B4/B5 HTTP 路由
│   └── run_service.py                        # Agent 运行与 B5 后台反思调度
└── frontend/src/
    ├── B4ModuleView.tsx                      # B4 页面入口
    ├── B4ObservationPanel.tsx                # 真实模型调用观察
    ├── B4DemoPanel.tsx                       # 模型/解析器协议演示
    ├── B4ViewShared.tsx                      # B4 公共展示组件
    ├── B5ModuleView.tsx                      # B5 快照与召回演示
    └── moduleViewUtils.ts                    # 模块页公共展示函数
```

### 3.2 接口边界

| 类型 | 来源 / 去向 | 数据格式 | 说明 |
|---|---|---|---|
| B4 输入 | B1 → B4 | `messages: list`、`tools_schema: list`、模型配置、可选图片 | B1 决定阶段和业务目标，B4 负责生成 |
| B4 输出 | B4 → B1 | `{ai_message, status, error, raw_text, prompt_messages}` | `ai_message` 含 content、tool_calls、control 和可选 agent_step |
| B4 阶段输出 | B1 → B4 → B1 | JSON object | `generate_json_object()` 为 B1 planning/observation 等阶段提供结构化结果 |
| B5 legacy 输入 | CLI / B1 → B5 | memory ids、global 开关、query 或保存输入文件 | 完成课程基础文档记忆接口 |
| B5 事实输入 | 后端 → B5 | conversation、message、tool step、完成轮次 trace | 原始记录先写入 SQLite，再进行反思 |
| B5 召回输入 | B1 / 后端 → B5 | conversation id、当前问题、历史消息、模型配置 | 构造候选、向量、rerank 和来源证据 |
| B5 输出 | B5 → B1 | `workspace_memory_context` / `layered_memory_context` | B1 只消费上下文包，不进入 B5 内部算法 |
| 观察接口 | 后端 → 前端 | JSON | B4 calls/protocol tests；B5 memory snapshot/recall preview |

关键不变量：

- B4 返回标准消息但不执行 tool_calls；工具执行仍属于 B3/B2。
- B5 的摘要、任务和块均须可回查原始消息或工具步骤。
- B5 模型增强失败时允许降级，但失败状态必须保留，不能伪装成完整成功。
- 前端模块页不修改核心算法，只展示真实 API 和运行产物。

---

## 4. 基础要求实现与演示

### 4.1 基础功能说明

#### B4 基础功能

- 读取 `configs/model.yaml`，选择 `local`、`fastapi` 或 `qwen_api` 模型源；
- 接收 messages 与 tools schema，构造模型输入；
- 生成完整或流式模型输出；
- 将 raw text 解析为标准 AIMessage；
- 校验 content、tool_calls、control 和工具参数结构；
- 保存 raw output、AIMessage 和 LLM 调用日志。

#### B5 基础功能

- 读取 `configs/memory.yaml`；
- 根据 memory id 加载指定文档；
- 可选加载全局 memory；
- 按 `max_memory_chars` 限制上下文长度；
- 保存 messages、trace 和 final answer 为 Markdown memory；
- 更新 `memory_index.json` 并记录 memory 日志。

### 4.2 基础功能实现路径

| 文件 / 函数 | 作用 |
|---|---|
| `b4_local_agent_llm._llm_source()` | 解析模型来源并限制合法值 |
| `b4_local_agent_llm.generate_ai_message()` | B4 非流式统一入口 |
| `b4_local_agent_llm.stream_ai_message()` | B4 流式入口和最终解析 |
| `b4_local_agent_llm.parse_model_output()` | 将独立 raw text 解析为标准 AIMessage |
| `b4_local_agent_llm._write_generation_artifacts()` | 保存 raw output、AIMessage 和日志 |
| `b5_memory_parts.legacy.load_memory()` | 读取索引、选择文档、控制总字符数 |
| `b5_memory_parts.legacy.save_memory()` | 生成 Markdown memory、更新索引和日志 |
| `b5_memory.py` | B5 legacy CLI 参数校验与分派 |

```text
B4：messages + tools_schema + model.yaml
  → 模型生成
  → raw text
  → 协议解析与校验
  → AIMessage + 调用产物

B5 legacy：memory.yaml + memory ids / save input
  → 路径和索引校验
  → 文档读取或保存
  → selected/saved memory + 日志
```

### 4.3 基础功能输入格式与样例

#### B4 输入

| 字段 / 文件 | 类型 | 必需 | 说明 |
|---|---|---:|---|
| `messages` | JSON 消息数组 | 是 | 角色限于 system、user、assistant、tool |
| `tools_schema` | JSON 数组 | 是 | B3 生成或样例提供；无工具时可为空数组 |
| `model_config` | YAML 路径 | 是 | 模型来源、路径、生成参数和接口地址 |
| `mode` | `prompt_json` / `mock` | 是 | 正式模型演示使用 `prompt_json` |
| `artifact_dir` | 目录路径 | CLI 必需 | 保存原始输出和标准消息 |

样例文件：

| 样例 | 用途 |
|---|---|
| `data/messages/messages_no_tool.json` | 生成最终回答或 tool_calls |
| `data/messages/messages_with_tool.json` | 基于成功 ToolMessage 回答 |
| `data/messages/messages_with_error_tool.json` | 工具错误后的回答收束 |
| `data/messages/tools_schema_basic.json` | 基础工具说明 |

#### B5 输入

| 字段 / 参数 | 类型 | 必需 | 说明 |
|---|---|---:|---|
| `--config` | YAML 路径 | 是 | `configs/memory.yaml` |
| `--select_memory_ids` | 字符串列表 | 查找可选 | 指定 memory id |
| `--use_global_memory` | 布尔值 | 查找可选 | 是否加载全局 memory |
| `--query` | 字符串 | 否 | 记录本次查找主题 |
| `--save_type` | conversation / global | 保存必需 | 与保存输入内的类型一致 |
| `--save_input_path` | JSON 路径 | 保存必需 | 包含 messages、trace、answer 相对路径 |

### 4.4 基础功能演示命令

以下命令从 `code/` 目录执行：

```bash
cd code

# B4：真实 prompt_json，需要当前模型服务可用
python b4_local_agent_llm.py \
  --model_config ../configs/model.yaml \
  --messages ../data/messages/messages_no_tool.json \
  --tools_schema ../data/messages/tools_schema_basic.json \
  --mode prompt_json \
  --outdir ../outputs/B4_llm/no_tool_real

# B5：legacy memory 查找
python b5_memory.py \
  --config ../configs/memory.yaml \
  --select_memory_ids mem_conversation_conv_000 \
  --use_global_memory true \
  --query "Agent 如何调用工具？" \
  --outdir ../outputs/B5_memory

# B5：legacy conversation memory 保存
python b5_memory.py \
  --config ../configs/memory.yaml \
  --save_type conversation \
  --save_input_path ../data/memory_inputs/memory_save_input.json \
  --outdir ../outputs/B5_memory
```

观察重点：

- B4 的 `raw_model_output.json` 中 `status`、`error`、`raw_text` 和 `prompt_messages`；
- B4 的 `ai_message.json` 是否满足标准结构；
- B5 的 memory id、文档路径、字符数、截断状态和错误数组；
- 保存操作是否生成 Markdown 文档并同步更新索引。

### 4.5 基础功能输出格式

| 输出 | 格式 | 说明 |
|---|---|---|
| `raw_model_output.json` | JSON | B4 模型来源、raw text、prompt、解析候选、状态和错误 |
| `ai_message.json` | JSON | B4 标准 AIMessage |
| `llm_run_log.jsonl` | JSONL | B4 每次生成的状态和产物路径 |
| `selected_memory.json` | JSON | B5 已选文档、字符统计、截断和错误 |
| `saved_memory.json` | JSON | B5 新 memory id、类型和文档路径 |
| `memory_log.jsonl` | JSONL | B5 查找和保存记录 |
| `memory_index.json` | JSON | legacy memory 元信息索引 |

### 4.6 基础功能结果截图

本 README 不放置虚构或旧环境截图。实际演示完成后应补充：

1. B4 `raw_model_output.json` 与 `ai_message.json` 对照；
2. B4 生成 tool_calls、接收 ToolMessage 后生成最终 content；
3. B5 `selected_memory.json` 与保存后的 Markdown/索引；
4. 截图旁标注 commit、模型源、模型名和运行时间。

---

## 5. 进阶要求实现与演示

### 5.1 选择的进阶要求

#### B4 进阶要求对应

| 课程进阶要求 | 状态 | 对应实现 | 说明 |
|---|---|---|---|
| 单轮多个 tool_calls / 多 ToolMessage | 已实现协议支持 | `parse_model_output()`、`generate_ai_message()` | B4 可解析多个调用并接收含多个工具结果的 messages；执行仍由 B1/B3 完成。 |
| Plan-and-Execute | 项目层实现 | B1 workspace + B4 三类生成接口 | 规划和状态机属于 B1，B4 只提供阶段结构化生成。 |
| 按任务切换不同本地模型 | 部分完成 | `configs/model.yaml`、`_llm_source()` | 支持手动切换来源和模型，未实现按任务自动路由。 |
| 内置 tools schema 与 prompt 注入对比 | 未完成 | 当前主线为 `prompt_json` | 尚无原生工具协议对照实现和结果。 |
| 不同模型成功率与 token 对比 | 未完成 | 暂无批量评测 | 尚无固定样例集和 token 汇总。 |

#### B5 进阶要求对应

| 课程进阶要求 | 状态 | 对应实现 | 说明 |
|---|---|---|---|
| 关键词检索排序与 top-k | 已在 SQLite 主线实现 | `retrieval.py`、`text_utils.py` | 综合字段、文本、工具、任务和时间信号排序。 |
| 长度管理与自动摘要 | 已实现 | `reflection.py`、`retrieval.py` | 近期原文、轮摘要、3–8 轮记忆块和字符预算。 |
| Memory 更新与冲突管理 | 部分完成 | task memory 更新 | 支持任务状态与内容更新，尚无任意文档的显式重复/补充/冲突合并。 |
| 向量检索 | 已实现，可降级 | `vector_retrieval.py` | embedding 缓存、cosine 相似度和加权分数。 |
| 错误 Memory 影响分析 | 未完成 | 已有来源与召回日志 | 尚未形成正确/错误 memory 对照实验。 |

### 5.2 进阶功能一：B4 多模型源、流式与协议容错

#### 功能说明

B4 使用统一接口连接三类模型源，并在非流式与流式路径结束后执行相同的 AIMessage 校验。模型输出可能出现参数字段别名、Markdown 尾标、纯文本或局部 JSON；解析器只修复能够确定语义的偏差：

- 将 tool call 的 `parameters` / `arguments` 归一化为 `args`；
- 保留 content 与 tool_calls 共存的合法消息；
- 恢复可确认的纯文本 content 或尾部 Markdown 标记；
- 拒绝 content 和 tool_calls 同时为空的无效消息；
- 无法可靠解析时返回 `status=error` 和 fallback content，而不是伪造工具调用。

#### 实现路径

| 文件 / 函数 | 作用 |
|---|---|
| `_prompt_messages_for_model()` | 构造模型可用消息并处理图片输入 |
| `_prompt_json_generate()` | 本地 transformers 生成 |
| `_fastapi_prompt_json_generate()` | FastAPI / Qwen API 非流式生成 |
| `_fastapi_prompt_json_stream()` | 远端流式文本读取 |
| `_candidate_to_message()` | 归一化 content、tool_calls、control 和 agent_step |
| `_parse_model_output()` | 统一解析和错误路径 |
| `backend/b4_demo_service.py` | 6 个模型用例与 4 个解析器用例 |

```text
messages + schema
  → local / fastapi / qwen_api
  → raw text 或 token stream
  → 可确认的协议归一化
  → AIMessage 校验
  → B1
```

#### 演示方式与输出

启动完整系统后，在 B4 模块页运行协议演示。模型类用例会调用当前模型服务；解析器类用例只验证 B4 协议，不执行 B2/B3 工具。

| 输出 | 说明 |
|---|---|
| `outputs/backend_runs/b4_demo/<run_id>/b4_protocol_test_result.json` | 每个用例的 request、raw text、AIMessage、delta、错误和判定 |
| `summary.total/passed/failed` | 当次运行汇总；只能说明本次配置和服务状态 |
| `<case>_raw_model_output.json` / `<case>_ai_message.json` | 模型类用例的底层 B4 产物 |

### 5.3 进阶功能二：SQLite 分层记忆与后台反思

#### 功能说明

浏览器多轮会话以 SQLite 为事实存储。每个完成轮次先保存用户消息、助手消息和工具步骤，再异步生成定位信息：

- **Turn**：把用户/助手消息和工具引用组织为一轮；
- **Turn Summary**：主题、关键词、事实、决定、纠正、偏好和任务相关度；
- **Task Memory**：前台、暂停、完成或放弃任务的目标、进度和决定；
- **Memory Block**：根据任务/主题边界、轮数和文本长度聚合 3–8 轮；
- **Source Evidence**：摘要和块保留原始 message/tool step id。

后端在最终回答完成后使用后台线程调度反思，取消的回答不进入完成态反思。模型反思失败时使用中性 fallback，并保留原始事实。

#### 实现路径

| 文件 / 函数 | 作用 |
|---|---|
| `conversation_store.init_store()` | 创建并迁移 SQLite 表结构 |
| `reflection.record_completed_turn_memory()` | 完成轮次的反思总入口 |
| `_coerce_memory_decision()` | 校验模型生成的标签、分数、摘要和任务动作 |
| `_apply_task_memory_decision()` | 更新任务记忆状态 |
| `_maybe_create_memory_block()` | 判断边界并创建多轮记忆块 |
| `backend/run_service.schedule_completed_turn_memory()` | 后台调度反思，不阻塞最终回答 |

```text
原始 Messages + Tool Steps + B1 Trace
  → SQLite 事实表
  → 结构化反思
  → Turn Summary / Task Memory
  → 满足边界时形成 Memory Block
  → 后续召回
```

#### 演示方式与输出

在同一浏览器会话中完成多轮任务，等待后台反思后打开 B5 页面。少量对话不一定立即形成记忆块，演示时应同时核对后台状态和块形成原因。

| 输出 / 数据 | 说明 |
|---|---|
| `memory/conversation_store.sqlite3` | 事实表和定位表 |
| `turn_summaries` | 轮摘要、标签和来源 id |
| `task_memories` | 任务状态、目标、进度和决定 |
| `memory_blocks` | 多轮记忆块及形成边界 |
| B5 memory snapshot API | 前端观察使用的结构化快照 |

### 5.4 进阶功能三：向量召回、LLM Rerank 与来源回查

#### 功能说明

每轮新问题开始前，B5 保留最近四轮原文，并从更早的 memory blocks 和 turns 中检索候选。召回包含三层：

1. **非向量评分**：文本相似度、字段重合、工具信号、任务相关度、长期价值和时间新近度；
2. **向量补分**：调用 `/embeddings`，缓存候选文本向量并计算 cosine 相似度；
3. **受约束 rerank**：LLM 只能从已有候选 id 中选择和排序，不能创造不存在的记忆。

向量或 rerank 不可用时，系统保留错误状态并回退到已有排序。最终默认最多选择 3 个块、5 个轮次，并加载对应原始消息和工具步骤；上下文受 `max_memory_chars` 限制。

#### 实现路径

| 文件 / 函数 | 作用 |
|---|---|
| `retrieval.build_layered_memory_context()` | 完整候选、评分、向量、rerank、来源和日志流程 |
| `retrieval.prepare_workspace_memory_context()` | 为 B1 生成精简上下文；异常时降级到近期原文 |
| `text_utils._score_block_detail()` / `_score_turn_detail()` | 非向量评分和细项 |
| `vector_retrieval.apply_vector_scores()` | embedding、缓存、相似度和权重 |
| `rerank.rerank_memory_candidates()` | 候选 id 约束和无效 id 记录 |
| `text_utils._build_memory_context_text()` | 按字符预算组装最终上下文 |

#### 演示方式与输出

在长对话后使用 B5 召回预览输入一个与早期约定相关的问题，不能只看最终摘要，应同时检查：

- `recalled_blocks`、`recalled_turns` 和分数细项；
- `vector_retrieval` 与 `llm_rerank` 状态；
- `source_messages` 和 `source_tool_steps`；
- retrieval log 中的候选 id、选择 id 和降级原因。

### 5.5 进阶功能四：B4/B5 可观察性页面

B4 观察页读取当前会话的真实模型调用，展示模型源、调用分类、prompt、raw output 和标准 AIMessage，并区分 Agent 主链路、B5 记忆辅助调用和独立演示。

B5 页面展示近期原文、轮摘要、记忆块、任务、召回日志和来源证据，并提供真实 recall preview。页面没有虚构“点击即压缩”的能力：记忆反思仍遵循回答完成后的后台生命周期。

```bash
# 从项目根目录启动后，由浏览器完成观察与演示
python start_all.py
```

实际截图应在本人运行后补充，并同时保留对应后端 JSON 产物。

---

## 6. 与团队系统的集成说明

### 6.1 B4 与 B1/B3 的集成

```text
B1 阶段与 messages
        +
B3 tools_schema
        ↓
       B4 ──→ AIMessage / 阶段 JSON
        ↓
AIMessage.tool_calls
        ↓
   B1 → B3 → B2
        ↓
   ToolMessage → B1 → B4 → final content
```

B1 负责 Planning、Tool Calling、Observation 和 Answering 状态；B4 只通过 `generate_json_object()`、`generate_ai_message()` 和 `stream_ai_message()`提供模型结果。因此项目级 Plan-and-Execute 不能写成 B4 单模块独立完成。

### 6.2 B5 与后端/B1 的集成

```text
上一轮完成
  → 后端保存原始消息和 Tool Steps
  → B5 后台反思
  → 摘要 / 任务 / 记忆块

下一轮开始
  → B5 召回近期原文和较早候选
  → Workspace Memory Context
  → B1 Workspace
  → B4 决策
```

B1 不参与 B5 内部评分、向量或 rerank；B5 也不决定 Agent 下一步动作。双方通过明确的 memory context JSON 对接。

### 6.3 前端协作范围

| 范围 | 本人重点 | 团队协作 |
|---|---|---|
| B4 页面 | 调用列表、详情观察、协议演示、共享展示组件 | 与后端运行目录和 B1 调用分类联调 |
| B5 页面 | 会话快照、任务/块/摘要、召回预览和来源展示 | 与 B1 上下文格式及后端反思状态联调 |
| B2/B3 页面 | 展示页和公共 JSON/状态组件整理 | 与对应模块负责人核对样例和错误语义 |
| 公共前端 | 模块导航、API 类型、样式和通用展示逻辑 | 与主对话、上传、下载和会话状态共同维护 |

### 6.4 联调中的主要接口问题

| 问题 | 处理方式 | 当前结果 |
|---|---|---|
| 模型 tool call 参数字段不稳定 | 在 B4 将 `parameters` / `arguments` 归一化为 `args`，同时保留严格校验 | B3 接收统一结构；无法确认的损坏输入仍报错 |
| raw output、标准消息和最终回答难以区分 | B4 分别保存 raw record 与 AIMessage，并在观察页并列展示 | 可定位是模型问题、解析问题还是上层编排问题 |
| legacy memory 无法承担浏览器多轮事实存储 | 保留 legacy 基础接口，新增 SQLite 主路径 | 基础演示与实际系统职责分开 |
| 记忆反思阻塞用户响应 | 回答完成后由后台线程调度反思 | 最终回答不等待反思，但 B5 页面存在短暂最终一致性延迟 |
| 摘要是否可靠难以判断 | 保存 source message/tool step id，召回时加载原始来源 | 精确事实可回查，不只依赖摘要文本 |
| 向量和 rerank 依赖外部服务 | 分层记录状态，并在失败时回退非向量排名 | 模型增强不可用时不阻断主回答 |
| 前端演示与真实生命周期可能不一致 | B4 区分模型用例和解析器回放；B5 只保留真实召回 | 页面不把静态样例或伪操作表述为真实模块运行 |

### 6.5 关键设计取舍

- **协议层不替代决策层**：B4 只恢复语义明确的格式偏差，不用关键词规则替模型选择业务工具。
- **事实优先于摘要**：B5 先保存原始消息和工具步骤，再生成摘要、任务和记忆块。
- **增强能力必须可降级**：反思、embedding 和 rerank 提升质量，但不能成为主对话的单点故障。
- **演示必须可追溯**：B4/B5 页面展示真实 API 与产物；解析器回放、mock 和真实模型调用需要明确区分。
- **协作修改保持边界**：涉及 B1/B3 或公共前端的改动以接口适配为主，不把 B4/B5 业务扩散到其他模块。

---

## 7. 已知问题与后续改进

| 问题 | 当前原因 | 后续改进 |
|---|---|---|
| 默认 API 模型与课程本地模型不同 | 当前 `model.yaml` 默认使用 `qwen_api/qwen-plus` | 验收前按要求切换 Qwen3.5-4B，并记录模型、硬件和产物 |
| B4 没有按任务自动路由模型 | 模型来源由配置统一选择 | 增加显式、可解释的路由配置，避免关键词硬编码 |
| 原生 tools schema 与 prompt JSON 对照未完成 | 当前主线优先保证 prompt JSON 稳定 | 使用固定任务集比较格式合法率、工具成功率、耗时和 token |
| 跨模型成功率和 token 统计未完成 | 缺少统一批量评测入口 | 与 B1 批量 runner 对接，生成可复现实验报告 |
| B4 解析错误会返回 fallback content | 需要避免上层完全无响应，但仅看 content 容易忽略解析失败 | 调用方和验收必须同时检查 `status` 与 `error`，后续可统一错误展示 |
| legacy memory 与 SQLite 双路径增加理解成本 | 一条用于课程基础验收，一条用于浏览器主系统 | 在页面和报告中持续标注数据来源，避免混用结论 |
| 任意 Memory 冲突合并不完整 | 当前主要更新 task memory 状态 | 增加 duplicate/supplement/conflict 类型、来源对比和人工确认 |
| 错误 Memory 影响实验未完成 | 目前只有来源证据和召回日志 | 构造正确/错误 memory 对照，比较召回、回答和纠正策略 |
| 后台反思存在短暂延迟 | 为避免阻塞最终回答采用异步线程 | 前端增加 pending/success/error 状态和刷新提示；后续可使用任务队列 |
| 向量与 rerank 依赖模型服务 | API、网络或额度异常时只能降级 | 增加健康状态、离线 embedding 选项和降级统计 |
| 少量对话不一定形成 Memory Block | 记忆块需要满足主题、任务、轮数或长度边界 | 准备满足 3–8 轮边界的固定演示会话，并展示形成原因 |
| 个人运行截图和固定结果尚未纳入仓库 | 本文不使用旧环境或虚构截图 | 由本人按固定 commit 和配置运行后补充截图及对应 JSON 证据 |

---

## 参考入口

- 项目总览：[`README.md`](README.md)
- 个人 README 模板：[`docs/PERSONAL_README_TEMPLATE.md`](docs/PERSONAL_README_TEMPLATE.md)
- B 方向课程说明：[`docs/B方向_Agent智能体_说明文档.docx`](docs/B方向_Agent智能体_说明文档.docx)
