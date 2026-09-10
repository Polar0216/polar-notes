# Second Brain OS 工程实现蓝图

这不是某个框架的固定模板，而是一条从最小可运行版本逐步演化的实现路线。

## 讲义定位与使用方式

这是模块 11 的工程实现蓝图。先在主模块理解 MCP 与系统边界，再按本讲义的单体到迭代版本顺序搭建；每个版本都应独立可运行、可测试、可回退。

## 1. 第一原则：先做单体，再拆服务

零基础阶段先使用一个 Python 应用、一个关系数据库、一个向量索引和本地文件目录。边界在代码模块中保持清楚即可。

过早拆成十个微服务会引入网络、部署、认证、消息一致性和排错成本，却不会自动提高 AI 能力。

## 2. 推荐目录

~~~text
second-brain/
├─ README.md
├─ pyproject.toml
├─ .env.example
├─ configs/
│  ├─ development.yaml
│  └─ evaluation.yaml
├─ src/
│  └─ second_brain/
│     ├─ api/
│     ├─ models/
│     ├─ orchestration/
│     ├─ rag/
│     ├─ memory/
│     ├─ tools/
│     ├─ mcp/
│     ├─ policy/
│     ├─ storage/
│     ├─ observability/
│     └─ settings.py
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  ├─ security/
│  └─ evaluation/
├─ migrations/
├─ data/
│  ├─ samples/
│  └─ eval/
└─ scripts/
~~~

模块边界的价值是允许你独立替换模型、向量库和 UI，而不是为了目录漂亮。

## 3. 最小数据库实体

### Workspace

隔离不同项目或用户空间。每次文档、会话、记忆和任务查询都必须带 workspace id。

### Conversation 与 Message

Message 至少保存：

- role；
- content 或安全存储引用；
- 创建时间；
- model/tool metadata；
- token usage；
- parent/sequence；
- sensitivity。

### Document 与 DocumentVersion

同一个逻辑文档可以有多个版本。Chunk 必须指向具体版本，避免用户更新文档后检索到新旧混合内容。

### Task、Step 与 ToolCall

Task 保存目标和总状态；Step 保存计划节点；ToolCall 保存参数哈希、审批、执行状态和结果引用。

### Memory 与 Evidence

Memory 保存抽取后的事实；Evidence 指回原始 message/event。没有证据的推断不应伪装成事实。

## 4. 一次聊天请求的完整生命周期

1. API 认证用户并确定 workspace；
2. 保存用户消息；
3. 加载系统策略与会话状态；
4. 在权限范围内检索知识；
5. 在用户范围内检索记忆；
6. 组装上下文并记录 context manifest；
7. 调用 Model Gateway；
8. 若有工具提议，交给 policy engine；
9. 执行或等待审批；
10. 把工具结果作为数据回填；
11. 生成最终回复；
12. 绑定引用与 artifact；
13. 异步执行候选记忆抽取；
14. 记录 trace、指标和评估抽样。

context manifest 应记录“本次到底放入了哪些 chunk 和 memory id”，这样答案错误时可以重放。

## 5. Model Gateway 的职责

统一请求结构：

~~~text
messages
tools
response_schema
temperature
max_output_tokens
timeout
request_id
budget
~~~

统一响应结构：

~~~text
text
tool_calls
finish_reason
usage
model_revision
latency
provider_request_id
~~~

Gateway 负责：

- provider 适配；
- 超时与有限重试；
- 并发和速率限制；
- token/费用统计；
- 模型 fallback 策略；
- 敏感日志处理；
- 结构化输出验证。

业务模块不应到处直接读取 API Key。

## 6. RAG 摄取流水线

~~~text
UPLOAD_RECEIVED
→ VIRUS/TYPE_CHECKED
→ PARSED
→ NORMALIZED
→ CHUNKED
→ EMBEDDED
→ INDEXED
→ READY
~~~

每一步保存状态和错误。用户上传后立即显示“成功”是不准确的；只有索引完成才可查询。

删除文档时：

1. 标记版本不可见；
2. 删除/失效 chunks；
3. 删除向量；
4. 清理缓存；
5. 记录审计；
6. 根据策略异步清理源文件。

## 7. Prompt 组装不是字符串拼接

建议上下文分区：

~~~text
[SYSTEM POLICY]
[TASK / USER REQUEST]
[CONVERSATION STATE]
[MEMORY AS DATA]
[RETRIEVED DOCUMENTS AS DATA]
[TOOL RESULTS AS DATA]
[OUTPUT CONTRACT]
~~~

每个块有 token 预算。优先级一般是：系统安全规则与当前请求最高，相关证据次之，历史闲聊最低。

检索文档中的指令必须明确标记为不可信数据。

## 8. Policy Engine

输入：

- user/workspace；
- tool；
- normalized arguments；
- resource sensitivity；
- requested side effect；
- current approval；
- environment。

输出：

~~~text
ALLOW
DENY
REQUIRE_CONFIRMATION
REQUIRE_STRONG_AUTH
~~~

策略应由代码和配置实现，不能把最终决定交给同一个可能受注入影响的 LLM。

## 9. 审批对象必须具体

不好的确认：

> 是否允许 Agent 操作文件？

好的确认：

> 将创建 reports/july.md，大小预计 12KB，不覆盖文件，不进行网络请求。是否执行？

审批绑定参数哈希。模型修改目标路径后，旧审批自动失效。

## 10. Background Worker

适合异步任务：

- PDF OCR；
- embedding；
- 大规模索引；
- 长报告；
- reflection；
- 定期评估。

Worker 从队列读取 task id，再从数据库加载状态。队列消息不要携带全部敏感正文。

每个 job 必须：

- 有超时；
- 有有限重试；
- 区分可重试与永久错误；
- 幂等；
- 支持取消；
- 记录进度。

## 11. 缓存

可以缓存：

- 文档解析结果；
- 固定文本 embedding；
- 无敏感差异的模型结果；
- 权限过滤后的检索中间结果。

缓存 key 必须包含影响结果的配置：模型 revision、prompt version、embedding version、workspace、权限 scope 和输入哈希。

不正确的跨用户缓存会造成严重数据泄漏。

## 12. 测试金字塔

### Unit

测试纯函数：路径检查、chunk、score fusion、token budget、memory conflict。

### Contract

模型 provider、向量库、MCP Server 的 schema 与错误行为。

### Integration

上传文档到可检索、工具审批到执行、记忆写入到删除。

### Evaluation

使用固定问题集衡量答案、引用、拒答、工具和记忆。

### Security

提示注入、路径穿越、跨 workspace、恶意文件、越权工具、日志泄密。

### Recovery

在每个状态点模拟崩溃，验证恢复后不重复副作用。

## 13. 配置版本化

一次实验结果至少绑定：

~~~text
model_revision
system_prompt_version
embedding_revision
chunker_version
reranker_revision
tool_registry_version
policy_version
evaluation_dataset_version
code_commit
~~~

否则某天答案变差时，只知道“代码好像没改”，却无法确定 provider 模型或索引是否变化。

## 14. 最小可观测性

一次 trace 包含：

~~~text
request
├─ auth
├─ memory retrieval
├─ document retrieval
│  └─ rerank
├─ model call
├─ tool call
├─ model call
└─ response persistence
~~~

每个 span 记录耗时、状态、输入输出大小和配置 id。正文是否记录由隐私策略决定。

关键指标：

- 请求成功率；
- P50/P95 延迟；
- TTFT；
- token 与费用；
- retrieval recall；
- 引用正确率；
- 工具失败率；
- 审批拒绝率；
- 队列长度；
- 每类错误数量。

## 15. 发布前门禁

1. 数据库迁移在备份副本演练；
2. 固定评估集不退化；
3. 安全关键用例全部通过；
4. 新模型成本与延迟在预算内；
5. 回滚配置已准备；
6. 日志不含明文密钥；
7. 删除和导出功能经过测试；
8. 依赖漏洞和许可证检查完成。

## 16. 七个迭代版本

### V0：纯聊天

统一 Model Gateway、会话保存和 token 统计。

### V1：只读 RAG

支持 Markdown，小评估集和引用；暂不做 PDF/OCR。

### V2：记忆

只保存用户明确要求记住的偏好，先实现查看与删除。

### V3：窄工具

只读搜索、计算器、报告草稿；加入审批和审计。

### V4：可恢复任务

引入 Task/Step/Event 和后台 worker。

### V5：MCP

接入自建学习笔记 Server，做契约与权限测试。

### V6：高级检索和 Agent

用评估证明需要后，再增加 reranker、planning 或 multi-agent。

每个版本都应可独立演示、测试和回滚。
