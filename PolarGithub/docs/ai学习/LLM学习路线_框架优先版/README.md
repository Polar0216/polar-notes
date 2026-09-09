# 从整机到零件：LLM 框架优先学习路线

这是一套面向计算机初学者的 LLM 教程。它保留原版讲义的原理与实现深度，但改变学习顺序：先认识并运行一台完整的最小 GPT，再逐步拆解 Tokenizer、Embedding、Attention、MLP、损失函数和反向传播，最后进入预训练、推理、微调、RAG 与 Agent 工程。

原有教程位于 `../LLM学习路线/`，本版本不会替换或修改原文件。原版适合作为详细参考书，本文件夹适合作为第一次学习时的主线课程。

## 1. 课程始终围绕同一条主线

整套课程只反复研究下面这台机器：

```text
文本
  ↓
Tokenizer：文字变成 token id                         [B, T]
  ↓
Token Embedding + Position Embedding：id 变成向量    [B, T, C]
  ↓
Transformer Block × N
  ├─ Causal Self-Attention：不同位置交换信息          [B, T, C]
  └─ MLP：每个位置内部加工信息                        [B, T, C]
  ↓
Final LayerNorm                                      [B, T, C]
  ↓
LM Head：每个位置产生词表分数                         [B, T, V]
  ↓
Softmax：词表分数变成概率                             [B, T, V]
  ↓
选择下一个 token，接回输入并重复
```

训练只是在这条主线上增加一条反馈路径：

```text
logits [B, T, V]
  ↓ 与真实的下一个 token 比较
Cross Entropy Loss
  ↓
backward()：计算每个参数对错误的责任
  ↓
optimizer.step()：小幅修改所有参数
```

以后遇到任何术语，先在这张图中定位，再学习细节。

## 2. 三遍学习法

| 遍次 | 核心目标 | 暂时允许不懂的内容 | 完成标志 |
|---|---|---|---|
| 第一遍：搭建框架 | 看懂文本如何变成下一个 token；运行完整程序 | 矩阵推导、反向传播细节、优化器公式 | 能从输入到输出复述全流程并运行代码 |
| 第二遍：拆解机器 | 理解每个核心部件的输入、输出、形状与职责 | 大规模训练、量化和服务优化 | 能解释并修改 MiniGPT 的核心实现 |
| 第三遍：补齐工程 | 掌握数据、训练、评估、Checkpoint、推理与应用栈 | 分布式训练和前沿优化可继续后补 | 能完成一项可复现项目并写实验报告 |

第一遍不要因为某个公式没完全掌握而停住。先获得全局坐标，第二遍再回来解决公式与代码。

## 3. 推荐阅读顺序

### 3.1 第一遍：搭建完整认知框架

1. [00_LLM全局地图.md](00_LLM全局地图.md)
2. [01_第一次运行MiniGPT.md](01_第一次运行MiniGPT.md)
3. [02_训练与生成的完整闭环.md](02_训练与生成的完整闭环.md)

这一遍只要求能回答：输入是什么、每个部件大致做什么、输出是什么、训练和生成有何不同。

### 3.2 第二遍：逐层拆解核心部件

4. [03_张量_Linear_Softmax与损失.md](03_张量_Linear_Softmax与损失.md)
5. [04_Tokenizer_Embedding与位置.md](04_Tokenizer_Embedding与位置.md)
6. [05_Attention与TransformerBlock.md](05_Attention与TransformerBlock.md)
7. [06_梯度_反向传播与优化.md](06_梯度_反向传播与优化.md)
8. [07_完整MiniGPT实现导读.md](07_完整MiniGPT实现导读.md)

这一遍必须频繁运行代码、打印 `shape`，并把所有部件重新挂回全局地图。

### 3.3 第三遍：训练与工程实践

9. [08_数据_预训练_评估与Checkpoint.md](08_数据_预训练_评估与Checkpoint.md)
10. [09_推理_采样_KVCache_量化与部署.md](09_推理_采样_KVCache_量化与部署.md)
11. [10_微调_RAG_Agent与后续方向.md](10_微调_RAG_Agent与后续方向.md)
12. [11_项目阶梯与验收标准.md](11_项目阶梯与验收标准.md)

这一遍开始强调可复现性、评估、资源预算与项目交付。

## 4. 配套文件

| 文件 | 用途 |
|---|---|
| [code/mini_gpt_complete.py](code/mini_gpt_complete.py) | 单文件完整 MiniGPT，实现训练、验证、生成、保存和恢复 |
| [code/tiny_corpus.txt](code/tiny_corpus.txt) | 可直接训练的原创中文小语料 |
| [code/README.md](code/README.md) | Conda 环境、运行命令与常见报错 |
| [学习记录模板.md](学习记录模板.md) | 每章学习与实验记录模板 |

## 5. 全程固定的形状语言

下面这张表是查阅表，不是第一课需要一次背完的词汇表。第一次只使用 $B$、$T$、$C$：一批文本、每条文本的 token 数、每个 token 的特征数。走到输出层时再加入 $V$，走到多头 Attention 时再加入 $H$ 和 $D$。

| 字母 | 含义 | 示例 |
|---|---|---|
| $B$ | Batch size，一批样本数 | 4 |
| $T$ | Sequence length，序列长度 | 32 |
| $C$ | Channel 或 embedding dimension，向量宽度 | 64 |
| $V$ | Vocabulary size，词表大小 | 3000 |
| $H$ | Attention head 数量 | 4 |
| $D$ | 每个头的宽度，通常 $D=C/H$ | 16 |

必须养成先写形状再写代码的习惯：

```text
idx       [B, T]
x         [B, T, C]
q, k, v   [B, H, T, D]
attention [B, H, T, T]
logits    [B, T, V]
targets   [B, T]
loss      标量
```

形状应该被读成数据的地址结构，而不是一串字母。例如 `x[b,t,:]` 表示第 $b$ 条文本中第 $t$ 个 token 的完整 $C$ 维向量。第 3 章会从具体小数组开始逐个展开。

## 6. 每一章的使用方法

每个新概念按固定因果顺序出现：

1. 当前任务需要实现什么结果。
2. 没有该组件时，最直接的方法会遇到什么具体问题。
3. 为解决问题需要引入哪一种计算思想。
4. 这个思想对应的正式组件名称。
5. 用一个短句或极小数字完整运行一次。
6. 最后补充公式、Shape 与 PyTorch 实现。
7. 把组件放回 GPT 全局数据流。

这里借鉴 3Blue1Brown 的讲解取向：先观察 token 向量需要发生怎样的变化，再进入执行变化的矩阵。[GPT 结构可视化](https://www.3blue1brown.com/lessons/gpt/)和 [Attention 分步可视化](https://www.3blue1brown.com/lessons/attention/)可以与第 0、5 章对照观看。

类比只用于指出某个结构关系，并明确停止位置。例如 Query、Key、Value 可以辅助区分“匹配依据”和“传递内容”，但真实注意力头的行为由训练决定，不能把每个头固定解释成一个人为角色。

不要把“看懂文字”等同于“掌握”。能预测形状、解释错误、修改代码并复现实验，才算掌握。

## 7. 第一遍的取舍原则

第一遍需要掌握：

- GPT 是自回归语言模型，核心任务是预测下一个 token。
- Tokenizer、Embedding、Transformer Block、LM Head 和 Softmax 的位置。
- Attention 负责位置之间的信息交流，MLP 负责单个位置内部的信息加工。
- 训练使用正确答案计算损失并修改参数，生成没有正确答案，只能逐 token 循环。
- `[B,T] → [B,T,C] → [B,T,V]` 是最重要的形状主线。

第一遍暂时跳过：

- BPE 合并规则的完整手算。
- Softmax、LayerNorm、AdamW 的详细推导。
- 多头拆分与合并的全部矩阵细节。
- 显存公式、KV Cache、量化、分布式训练。
- LoRA、RAG、Agent 的工程细节。

跳过不等于删除。这些内容会在第二遍和第三遍回到正确的位置。

## 8. 原版讲义的查阅方式

当新教程提示“进入参考层”时，再查原版对应材料：

| 当前主题 | 原版详细参考 |
|---|---|
| Python、NumPy、数学 | `../LLM学习路线/01_Python_Git_数学基础.md` |
| 神经网络与 PyTorch | `../LLM学习路线/02_NumPy神经网络与PyTorch.md` |
| Attention 与 Transformer | `../LLM学习路线/03_Tokenizer_Attention_Transformer.md` 与 `03A` |
| MiniGPT 与预训练 | `../LLM学习路线/04_MiniGPT与预训练.md` 与 `04A` |
| 推理和部署 | `../LLM学习路线/05_推理量化与部署.md` 与 `05A` |
| 微调、RAG、Agent | 原版 `06` 至 `12` 模块 |

新教程负责保持主线，原版讲义负责提供更细的查阅材料。两者不是重复关系。

## 9. 完成整套课程后的能力边界

完成后应当能够：

- 从文本开始解释 GPT 的完整前向、训练和生成流程。
- 阅读一个教育用途的 GPT 实现并标注所有主要形状。
- 在 Conda 环境中训练、保存、恢复并生成文本。
- 用小数据进行过拟合测试、验证集评估和故障定位。
- 解释 Attention、MLP、残差、LayerNorm、交叉熵和优化器的职责。
- 区分预训练、SFT、LoRA、RAG、Tool Calling 与 Agent。
- 根据目标选择继续深入模型训练、推理部署或 LLM 应用工程。

这套课程不会让小型字符模型拥有 ChatGPT 的能力。它的作用是把现代 LLM 的完整骨架缩小到个人电脑可以观察、修改和验证的尺度。
