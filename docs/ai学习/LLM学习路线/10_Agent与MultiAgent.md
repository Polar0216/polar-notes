# 模块 10：Agent、Browser/Coding Agent 与 Multi-Agent

## 目标

从受控 ReAct 循环进化到状态机式规划 Agent；理解浏览器和代码 Agent 的环境、反馈、安全与评估；只在任务可分解且收益可测时使用 Multi-Agent。

> 一个任务如何经历契约、计划、执行、校验、崩溃恢复和最终停止，见 [从 Tool Calling 到可恢复 Agent：完整案例](08A_从Tool到Agent完整案例.md)。

## 学习导航

先实现受控的单 Agent 状态循环，再学习规划、浏览器、编码和协作；先完成验收与恢复测试，最后再做文末 Project。多 Agent 必须与单 Agent 基线比较。

## 1. 什么是 Agent

Agent 是一个循环系统：模型根据目标和状态决定下一动作，环境执行并返回观察，系统更新状态，直到完成、失败或需要人类。

```text
Goal + State
→ Decide
→ Action/Tool
→ Observation
→ Update State
→ Stop / Continue / Ask Human
```

“一次 LLM 调用接几个工具”也能做很多事，不必为了 Agent 而 Agent。

## 2. ReAct

ReAct 交替进行推理与行动。实际产品不应依赖或展示自由形式的隐藏“Thought”；更稳妥的是让模型输出结构化的 `action`、`arguments`、`brief rationale`/状态字段，应用保存可审计事件。

状态示例：

```python
state = {
    "goal": "...",
    "messages": [],
    "observations": [],
    "artifacts": [],
    "steps": 0,
    "budget": {"max_steps": 12, "max_cost": 1.0},
    "status": "running",
}
```

必须有终止条件：任务完成、最大步数、时间/成本上限、重复动作检测、不可恢复错误、等待用户确认。

## 3. Planning Agent

规划将大目标拆成有依赖和验收条件的子任务。计划不是圣旨：观察变化后应重规划。

一个好 step 包含：目标、输入、允许工具、完成条件、输出产物、依赖和风险。比如“搜索资料”不是可验收步骤；“找到 3 个一手来源并保存标题/日期/链接”才是。

计划执行模式：

- Plan-and-execute：先整体计划再执行，清晰但计划易过时。
- Interleaved：每步后重新判断，更适应环境但调用多。
- DAG/workflow：固定依赖，可并行且容易恢复，适合稳定流程。

## 4. 状态机比无限循环更可靠

```text
RECEIVED → PLANNING → EXECUTING → VERIFYING → DONE
                         ↓             ↓
                    NEED_APPROVAL    REPAIR
                         ↓             ↓
                       PAUSED ←───────┘
```

每个 transition 要有明确条件，状态持久化后进程崩溃可恢复。工具结果用事件追加，避免在重试时重复执行副作用。

## 5. Reflection / Reviewer

Reviewer 不应只说“看起来很好”，而应基于验收标准和可观察证据：测试是否通过、引用是否支持、文件是否存在、页面状态是否符合。自我批评模型与生成模型共享盲点；能用确定性测试就先用测试。

限制修复轮数，记录每次修改原因，若同一错误重复出现则升级给人类。

## 6. Browser Agent

浏览器 Agent 的动作空间包括 navigate、click、type、scroll、extract、download。网页是动态且敌对的环境：DOM 会变化、按钮同名、弹窗、登录、验证码、提示注入。

工程要点：

- 使用可访问性树/稳定 selector，避免只靠屏幕坐标。
- 每次动作后重新观察，不假设页面未变。
- 导航限定允许域；下载视为不可信文件。
- 登录凭证由专门 secret 管理器填充，不进入模型上下文。
- 提交表单、购买、发布前人工确认。
- 记录页面 URL、动作和截图/DOM 摘要用于复现。

评估用本地可控网站或专门 benchmark，避免对真实网站产生副作用。

## 7. Coding Agent

Coding Agent 的核心循环：理解仓库→搜索定位→小范围修改→运行测试/静态检查→读取错误→修复→总结 diff。

需要的边界：

- 工作区范围和文件读写权限。
- 依赖安装与网络权限。
- 命令 allowlist/审批。
- 禁止读取/回显 secrets。
- 保留用户未提交改动，不做破坏性 Git 操作。
- 测试命令超时和资源限制。

上下文选择比“把整个仓库塞进去”更重要。先读目录、README、配置和符号引用，再取相关文件与测试。

## 8. Multi-Agent 什么时候值得

适合：子任务真正可并行、需要不同工具/上下文、角色间产物接口明确。未必适合：小任务、强顺序依赖、大家共享同一盲点、通信成本超过工作量。

`Planner/Coder/Reviewer/Tester` 四个角色不是自动带来质量。每个角色要有：输入契约、输出 schema、权限、停止条件、共享工件位置和冲突解决规则。

```text
Planner  → plan.json
Coder    → patch + implementation notes
Tester   → test-results.json
Reviewer → findings.json
Coordinator 根据证据决定 done / repair / human
```

Tester 应运行测试，不只是让另一个 LLM“扮演测试员”。Reviewer 不应未经授权直接覆盖 Coder 的文件。

## 9. 通信与共享状态

常见拓扑：

- Supervisor：中央协调，易治理但可能成为瓶颈。
- Handoff：Agent 之间移交，灵活但难追踪。
- Blackboard：共享任务板/工件，适合异步但需并发控制。
- Debate：多候选互评，成本高且不保证正确。

消息应传结论、证据和工件引用，而不是复制全部对话。共享文件要版本化，写入采用锁/原子提交，防止互相覆盖。

## 10. Agent 评估

- Task success rate：真实完成率。
- Step efficiency：成功所需步数/工具调用数。
- Cost/latency：总 token、工具费用、墙钟时间。
- Recovery：工具失败、页面变化、测试失败后能否恢复。
- Safety：越权动作、未确认副作用、敏感信息泄露。
- Reproducibility：相同初始状态的方差。

建立轨迹回放：保存每步状态、动作、观察、耗时和结果。失败分类比单一总分更有助于改进。

## 11. 框架学习顺序

先手写 100～200 行状态循环，再看 LangGraph 的状态图和持久化；然后选一个 AutoGen/CrewAI/CAMEL 比较抽象。OpenHands、Browser Use 属于大型应用，先沿一条任务轨迹寻找入口、状态、工具、sandbox 和事件流，不要逐文件通读。

## 验收与复盘

- 所有循环有步数/时间/成本和重复检测。
- 可从持久化状态恢复而不重复副作用。
- Browser/Coding Agent 在隔离环境运行并有人工确认点。
- Multi-Agent 有单 Agent 基线和量化收益。
- 轨迹日志足以解释一次失败为何发生。

## 课后 Project

### Project 21：ReAct Agent

用计算器、搜索模拟器和笔记工具完成 30 个任务。结构化动作，最大 10 步，检测重复调用，有明确 `final/need_user/failed` 状态。

### Project 22：Planning Agent

实现计划 schema、依赖、完成条件和重规划。故意让一个工具失败，验证它会换方案或请求帮助，而不是死循环。

### Project 23：Browser Agent

在本地测试站点完成查找、筛选、填写但不最终提交。加入恶意页面文本，验证不会执行越权指令。

### Project 24：Coding Agent

在专用练习仓库完成 10 个 issue；每次必须先运行基线测试，修改后运行相关测试，再给 diff 与风险。与“直接把问题交给模型一次回答”作成功率对照。

### Project 25：Multi-Agent

让 Planner/Coder/Tester/Reviewer 协作实现一个小功能。对照单 Agent，比较成功率、成本、耗时、修改冲突和人工介入次数。如果没有提升，结论应是“此任务不需要多 Agent”，这同样是正确实验。
