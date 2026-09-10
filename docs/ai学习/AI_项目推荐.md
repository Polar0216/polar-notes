> 本路线的零基础配套教材、练习与验收标准见 [`LLM学习路线/README.md`](LLM学习路线/README.md)。

这个问题我很喜欢，因为**这其实就是一条完整的 LLM Engineer 成长路线**。

如果让我重新规划，我不会按照「Prompt→RAG→Agent」这种网上教程来，而是按照**LLM 一个产品是怎么诞生的**来学。

---

# 总目标

最终完成一个自己的项目：

> **Personal AI OS（个人 AI 操作系统）**

它拥有：

- 自己训练/微调的模型（或者至少懂整个流程）
- 长短期记忆
- Tool Calling
- Browser Agent
- Code Agent
- RAG
- MCP
- 多 Agent 协作
- 长上下文管理
- GUI
- 本地部署

在这个过程中，你会把 GitHub 上几乎所有主流 LLM 技术都学到。

---

# 第一阶段：Python + 深度学习基础（2~3 周）

> **目标：以后看到 Transformer 代码不会害怕。**

项目：

### Project 1：NumPy 实现神经网络

自己写

```
Linear
ReLU
Softmax
CrossEntropy
SGD
```

不要调用 torch.nn。

GitHub：

- d2l-ai
- tinygrad（后面可以看）

---

### Project 2：PyTorch 重写 MLP

实现

```
MNIST
训练
推理
保存模型
```

学习：

- Dataset
- DataLoader
- Module
- Optimizer

---

# 第二阶段：Transformer 从零实现（3~4 周）

> **目标：真正理解 GPT。**

项目：

## Project 3：Tokenizer

自己实现

```
Character Tokenizer

↓

Word Tokenizer

↓

BPE

↓

SentencePiece思想
```

最后做

```
train_tokenizer.py
```

---

## Project 4：Attention 可视化

自己实现

```
QKV

↓

Attention Score

↓

Mask

↓

Output
```

画热力图。

---

## Project 5：MiniGPT

GitHub：

- minGPT
- nanoGPT

目标：

训练一个几十 M 参数的小 GPT。

最后

```
输入：

你好

输出：

你好，很高兴……
```

---

# 第三阶段：模型训练（2 周）

## Project 6：Pretraining

数据：

Wikipedia

小说

自己的笔记

学习：

```
Dataset

↓

Packing

↓

Training

↓

Loss

↓

Checkpoint
```

---

## Project 7：Inference Engine

学习：

```
Temperature

Top-k

Top-p

Beam Search

KV Cache
```

自己实现采样。

---

# 第四阶段：微调（2 周）

## Project 8：LoRA

GitHub：

LLaMA Factory

目标：

微调

```
Qwen

Llama

Gemma
```

做一个

```
数学老师模型
```

---

## Project 9：QLoRA

理解

```
4bit

8bit

16bit

NF4

Quantization
```

为什么能省显存。

---

# 第五阶段：RAG（2~3 周）

## Project 10：最简单 RAG

流程

```
PDF

↓

Chunk

↓

Embedding

↓

Vector DB

↓

TopK

↓

Prompt

↓

LLM
```

GitHub：

LangChain RAG

---

## Project 11：高级 RAG

加入

```
Query Rewrite

Hybrid Search

Parent Document

ReRanking

Multi Query

Context Compression
```

---

## Project 12：GraphRAG

知识图谱。

这个很有意思。

---

# 第六阶段：Tool Calling（1~2 周）

## Project 13：Function Calling

例如

```
天气

计算器

Python

Shell
```

LLM

↓

自动调用。

---

## Project 14：Code Interpreter

实现

```
LLM

↓

生成Python

↓

运行

↓

返回结果

↓

继续修改
```

ChatGPT 就是这么干的。

---

# 第七阶段：Memory（重点）

我建议你花很多时间。

因为这是未来。

---

## Project 15：Conversation Memory

就是

```
最近10轮
```

---

## Project 16：Summary Memory

超过长度

↓

自动总结。

---

## Project 17：Vector Memory

聊天内容

↓

Embedding

↓

向量数据库

↓

召回

---

## Project 18：Long-term Memory

例如

```
用户喜欢机械

用户会Python

用户正在考研
```

建立 Profile。

---

## Project 19：Memory Importance

Google Generative Agents 那套。

记忆：

```
Importance

Recency

Relevance
```

三因素排序。

---

## Project 20：Memory Reflection

每天自动总结：

```
今天发生了什么

↓

形成长期记忆
```

这个特别酷。

---

# 第八阶段：Agent

## Project 21：ReAct Agent

```
Thought

↓

Action

↓

Observation

↓

Thought
```

---

## Project 22：Planning Agent

把

```
写论文
```

拆成

```
搜索

阅读

总结

写作
```

---

## Project 23：Browser Agent

GitHub：

Browser Use

Skyvern

自动点网页。

---

## Project 24：Coding Agent

类似

OpenHands

Continue

可以修改整个项目。

---

# 第九阶段：Multi-Agent

## Project 25：

四个 Agent：

```
Planner

Coder

Reviewer

Tester
```

一起完成任务。

GitHub：

AutoGen

CrewAI

CAMEL

---

# 第十阶段：MCP

Project 26：

自己写 MCP Server。

例如

```
SolidWorks

↓

MCP

↓

LLM
```

以后工业软件都会这样。

---

# 第十一阶段：AI OS

最终项目。

例如

```
聊天

↓

Planner

↓

Memory

↓

Tool

↓

Browser

↓

Python

↓

Code

↓

Calendar

↓

Email
```

真正形成一个

AI Operating System。

---

# 第十二阶段：机器人

如果以后偏机械。

Project 27：

Vision + LLM。

```
摄像头

↓

Qwen VL

↓

Planner

↓

ROS2

↓

MoveIt

↓

机械臂
```

这几年特别火。

---

# 推荐的 GitHub（按学习顺序）

## 入门

⭐⭐⭐⭐⭐

```
tinygrad
```

```
minGPT
```

```
nanoGPT
```

```
LLMs-from-scratch
```

---

## 推理

```
llama.cpp
```

```
vLLM
```

```
SGLang
```

---

## 微调

```
LLaMA Factory
```

```
Unsloth
```

---

## RAG

```
RAG From Scratch
```

```
RAG Techniques
```

```
GraphRAG
```

---

## Agent

```
LangGraph
```

```
OpenHands
```

```
OpenManus
```

```
Open Interpreter
```

```
Browser Use
```

---

## Memory

```
Mem0
```

```
Zep
```

```
Letta（原 MemGPT）
```

这是我非常建议重点看的三个项目，它们代表了目前长期记忆系统的三种典型思路。

---

## AI Coding

```
Continue
```

```
Aider
```

```
Cline
```

---

## 多 Agent

```
AutoGen
```

```
CrewAI
```

```
CAMEL
```

---

# 如果让我重新学一遍（推荐顺序）

```
Python
    ↓
PyTorch
    ↓
Tokenizer
    ↓
Transformer
    ↓
MiniGPT
    ↓
Pretrain
    ↓
Inference（KV Cache / Sampling）
    ↓
LoRA / QLoRA
    ↓
RAG（基础 → 高级 → GraphRAG）
    ↓
Tool Calling
    ↓
Memory（短期 → 总结 → 长期 → 反思）
    ↓
Agent（ReAct → Planning）
    ↓
Browser Agent
    ↓
Coding Agent
    ↓
Multi-Agent
    ↓
MCP
    ↓
AI OS
    ↓
Vision-Language Model
    ↓
Embodied AI（机器人）
```

---

## 我还建议增加一个“毕业项目”，把前面所有内容串起来

**项目名：Second Brain OS（第二大脑）**

它不是简单聊天，而是一个完整的 AI 系统：

- **LLM**：Qwen 或 Llama（本地或 API）。
- **记忆层**：短期上下文 + Summary Memory + 向量记忆 + 用户画像 + Reflection。
- **知识层**：RAG（PDF、笔记、代码仓库）。
- **工具层**：Python、Shell、浏览器、Git、日历等 Tool Calling。
- **Agent 层**：Planner、Executor、Reviewer 多 Agent 协作。
- **界面层**：Web UI 或桌面端。
- **扩展层**：MCP 接入更多软件，未来还能连接 ROS2 和机械臂。

**这个项目最大的价值不是功能，而是它会逼着你把前面学过的所有知识真正整合到一起。**

---

### 最后给你一个建议

**不要把 GitHub 当成“代码仓库”，而要把它当成“论文”。**

对于每个项目，建议按固定流程学习：

1. **先回答三个问题**
   - 它解决了什么问题？
   - 为什么以前的方法不够？
   - 它的核心创新是什么？
2. **再读源码**
   - 画出模块图。
   - 理清数据流。
   - 找到最核心的 3～5 个文件。
3. **最后自己复刻一个 Mini 版**
   - 不求功能完整，只保留核心思想。
   - 每复刻一个项目，都写一篇学习笔记，总结设计取舍。

这样一年下来，你收获的不只是会用框架，而是真正具备阅读和设计 LLM 系统的能力。
