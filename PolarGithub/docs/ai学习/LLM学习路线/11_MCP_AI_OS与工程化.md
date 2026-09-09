# 模块 11：MCP、Second Brain OS 与工程化

## 目标

理解 MCP 的协议边界，自己实现一个只读资源与受控工具 Server；把模型、RAG、Memory、Tools、Agent、UI 整合成可测试、可恢复、可观测的 Second Brain OS。

> 推荐目录、数据实体、请求生命周期、Policy Engine、后台任务和测试门禁见 [Second Brain OS 工程实现蓝图](11A_SecondBrain工程实现蓝图.md)。

## 学习导航

先明确 MCP 协议与安全边界，再阅读工程蓝图并完成测试门禁；通过验收与复盘后，最后实施文末的 MCP 与毕业 Project。

## 1. MCP 解决的不是模型能力，而是连接标准化

如果每个 AI 应用都为文件、数据库、Git、工业软件写一套私有连接器，会产生大量重复集成。MCP（Model Context Protocol）规定客户端和服务器如何发现能力、交换请求与结果。

```text
Host（聊天/IDE/AI OS）
  └─ MCP Client（管理连接与会话）
       └─ MCP Server（暴露资源/工具/提示）
            └─ 本地文件、数据库、SolidWorks API、业务系统
```

MCP 不会自动让工具安全，也不等于 Agent。协议提供连接方式；Host 仍负责授权、确认、隔离、上下文选择和模型调用。

## 2. 三类核心能力

### Resources

可读取的上下文数据，通常由应用选择/用户浏览，例如文件、文档、数据库记录。语义偏“读数据”，使用 URI 标识；可有列表、模板、订阅等能力（具体以当前规范为准）。

### Tools

模型可提议调用的操作，带输入 schema 和结构化结果。可能有副作用，所以需要权限与确认。

### Prompts

服务器提供的可复用提示模板，通常由用户/Host 选择。Prompt 不是越过 Host 系统规则的高优先级指令。

判断例子：读取零件说明书是 resource；查询某零件实时属性可做 resource 或只读 tool；修改 CAD 尺寸是高风险 tool；“生成工艺审查问题清单”可做 prompt。

## 3. Transport 与生命周期

本地 Server 常用 stdio，由 Host 启动子进程，通过标准输入输出通信。远程服务使用规范支持的 HTTP transport。不要把调试日志写入 stdio 协议通道；日志写 stderr。

连接典型流程：初始化与能力协商→列举能力→读取/调用→通知/进度→关闭。客户端不能假设所有服务器支持所有可选能力，应根据协商结果工作。

SDK 与规范会演进，锁定依赖主版本，记录 server/protocol 版本，并为升级建立契约测试。具体 transport 名称和弃用项以官方最新规范为准。

## 4. 最小 MCP Server 的设计

不要先连接 SolidWorks。先实现一个“学习笔记 Server”：

- Resource：列出/读取允许目录下的 Markdown。
- Tool：按关键词搜索笔记（只读）。
- Tool：在草稿目录创建笔记（写操作，需 Host 确认）。
- Prompt：把一篇笔记转换为复习题。

伪代码（API 名称应按当前 Python SDK 文档调整）：

```python
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("study-notes")
ROOT = Path("./notes").resolve()

def safe_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("path outside allowed root")
    return path

@mcp.tool()
def search_notes(query: str, limit: int = 5) -> list[dict]:
    """Search study notes in the allowed root; does not modify files."""
    if not query.strip() or not 1 <= limit <= 20:
        raise ValueError("invalid query or limit")
    ...
```

关键不在装饰器，而在 `safe_path`、参数上限、输出大小、错误结构和审计。

## 5. MCP 安全清单

- Server 来源可信、版本固定、代码/权限经过审核。
- Host 展示服务器提供的工具名、描述和风险。
- 每个 Server 独立最小权限；文件根、网络域、账号 scope 受限。
- 本地 stdio Server 不继承不必要的环境变量和密钥。
- 远程连接使用认证、TLS，防止 token 泄露与 confused deputy。
- 工具输出、resource 内容都是不可信数据，防 prompt injection。
- 写操作具体确认，显示目标、差异和副作用。
- 防止工具名冲突；内部使用 server id + tool name。
- 日志记录 server/call id、耗时和状态，正文按敏感级别脱敏。

安装“万能 MCP Server”相当于安装有权限的软件，不应只因为仓库星数高就信任。

## 6. Second Brain OS 分层架构

```text
UI / API
   ↓
Conversation & Identity
   ↓
Orchestrator / State Machine
   ├─ Model Gateway
   ├─ RAG / Knowledge
   ├─ Memory
   ├─ Tool Registry / MCP Clients
   ├─ Policy & Approval
   └─ Evaluation / Observability
          ↓
Storage: metadata DB / vector index / object files / event log
```

各层职责：

- UI：展示流式回复、引用、工具计划、确认和记忆管理。
- Model Gateway：统一模型接口、超时、重试、token/成本统计。
- Orchestrator：保存任务状态，决定下一节点，不把所有逻辑塞进一个 prompt。
- Knowledge：文档摄取、版本、检索和引用。
- Memory：用户可查看/修改/删除的跨会话状态。
- Tool/MCP：发现能力、schema 验证、执行和审计。
- Policy：按用户、工具、数据敏感度做代码级授权。
- Observability：trace、metrics、评估和失败回放。

## 7. 数据模型先于漂亮 UI

建议最小实体：

```text
User / Workspace / Conversation / Message
Document / DocumentVersion / Chunk
Memory / MemoryEvidence
Task / Step / ToolCall / Approval
Artifact / EvaluationRun
```

所有对象有稳定 id、创建/更新时间、owner/workspace、来源和状态。删除文档时要能找到对应 chunks 和向量；删除用户时要处理记忆、日志和产物。

## 8. Model Gateway

不要让各模块直接调用不同模型 SDK。统一接口：

```python
class ModelResponse:
    text: str
    tool_calls: list
    usage: dict
    finish_reason: str
    model: str

def complete(messages, tools=None, timeout_s=60, request_id=None): ...
```

Gateway 负责模型选择、重试、速率限制、熔断、缓存（只缓存安全可复用请求）、观测和敏感字段处理。模型输出必须连同实际模型版本与采样参数记录。

## 9. 状态、队列与恢复

长任务不要绑在一个 HTTP 请求中。把任务写入持久化队列/数据库，worker 按 step 执行，UI 查询或订阅进度。每个 step 应尽量幂等；副作用记录 idempotency key。

崩溃恢复时：读最后已提交状态→确认是否有未决工具调用→查询外部系统结果→决定重试/补偿/人工介入。不能简单“从头再跑”。

## 10. Observability

三类信号：

- Logs：离散事件，带 request/task/tool id。
- Metrics：成功率、延迟、token、费用、检索 recall、队列深度。
- Traces：一次请求跨模型、检索、工具的完整链路。

保存 prompt/文档正文可能帮助排错，也可能产生隐私风险。按环境和敏感级别选择采样、脱敏和保留期限。至少记录配置哈希，保证能重放。

## 11. 评估门禁

建立版本化评估集，覆盖：普通问答、无答案、引用、记忆更新、工具选择、提示注入、权限越界、工具故障和长任务恢复。

每次更换模型、prompt、embedding、chunk、reranker 或 agent policy，都跑回归。部署门槛不仅是平均正确率，还包括：关键安全用例 100% 通过、成本/延迟不超过预算、失败可解释。

## 12. GUI 最小功能

第一版只需：

- 会话与流式输出；
- 引用可点击并显示原文；
- 工具调用卡片和确认按钮；
- 任务步骤与取消；
- 记忆查看/修改/删除；
- 文档上传状态和错误；
- 设置页（模型、预算、隐私）。

不要先做炫酷头像和动画。AI 系统最有价值的 UI 是把证据、状态和副作用讲清楚。

## 验收与复盘

- 能解释 MCP 的 Host/Client/Server 和 resources/tools/prompts。
- MCP Server 有契约、越权、输出上限和错误测试。
- AI OS 分层明确，数据可追溯，任务可恢复。
- 任何外部写操作都有 policy 和审批。
- 模型或检索配置变化会自动触发回归评估。

## 课后 Project

### Project 26：MCP Server

先做学习笔记 Server，编写独立 client 测试：能力发现、合法读取、路径穿越、超长参数、工具异常、并发和关闭。然后再把某工业软件 API 包成少量只读工具；写操作最后做，并要求确认和事务/撤销。

### 毕业 Project：Second Brain OS 里程碑

1. M0：单模型聊天 + 配置化 provider。
2. M1：Markdown/PDF 摄取、检索、带引用回答和评估集。
3. M2：短期/摘要/长期记忆，提供管理页面。
4. M3：三个受控工具 + 审批 + 审计。
5. M4：可恢复 Planner/Executor，不超过固定预算。
6. M5：MCP 接入笔记/日历的测试账号。
7. M6：部署、身份认证、备份、监控和回归门禁。
8. M7（可选）：Multi-Agent；只有单 Agent 基线证明需要才加入。

每个里程碑保持可运行，并打 Git tag。不要等“全部做完”才首次集成。
