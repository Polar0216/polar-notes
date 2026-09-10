# 第二遍之二：Tokenizer、Embedding 与位置信息

本章拆解全局图的入口：文字如何变成离散编号，离散编号如何变成可训练向量，以及模型如何区分相同 token 出现在不同位置。

## 1. Tokenizer 定义文字与模型之间的接口

神经网络只能接收数字，Tokenizer 负责两种方向的确定性映射：

```text
encode：文本 → token id
decode：token id → 文本
```

Tokenizer 不是模型内部“理解文字”的神经网络层。它通常在模型前处理文本，并决定：

- 词表包含哪些 token；
- 文本如何被切分；
- 每个 token 使用哪个整数 id；
- 特殊 token 如何表示；
- 未知内容如何处理。

模型的 `vocab_size` 必须与 Tokenizer 的词表大小一致。

## 2. 字符级 Tokenizer 的完整机制

配套代码使用字符级 Tokenizer：

```python
characters = sorted(set(text))
tokens = ["<unk>"] + characters
stoi = {token: index for index, token in enumerate(tokens)}
itos = {index: token for index, token in enumerate(tokens)}
```

编码：

```python
ids = [stoi.get(character, stoi["<unk>"]) for character in text]
```

解码：

```python
text = "".join(itos[index] for index in ids)
```

字符级方案的优点是实现透明、不会出现复杂分词。缺点是序列较长，中文、英文与标点的统计效率不高，也不代表真实大模型常用的 Tokenizer。

## 3. 子词 Tokenizer 的工程价值

真实 LLM 常使用 BPE、Unigram 或相近的子词方法，在“整词”和“单字符”之间折中：

```text
整词级：序列短，但词表巨大，生僻词难处理
字符级：词表小，但序列长，语义单元过碎
子词级：常见片段合并，生僻词仍可拆分
```

第一遍不需要手算 BPE。第二遍只需理解训练过程的大致目标：从基础符号开始，反复把语料中高频相邻片段合并为新 token，直到达到目标词表规模。

Tokenizer 属于数据合同。更换 Tokenizer 会改变 token id、序列长度与词表大小，原模型的 Embedding 和 LM Head 通常无法直接沿用。

## 4. 特殊 token 的语义约定

常见特殊 token 包括：

| token | 常见职责 |
|---|---|
| `<unk>` | 表示词表外内容 |
| `<bos>` | 序列开始 |
| `<eos>` | 序列结束或文档分隔 |
| `<pad>` | 把不同长度序列补齐 |

特殊 token 的名称本身不会自动赋予功能。它的实际含义来自训练数据的使用方式，以及模型配置和推理代码对相应 id 的一致处理。

配套字符模型只显式加入 `<unk>`，因为它使用连续语料随机切块，不需要 padding。真实批量对话训练通常还需要 EOS、PAD 与对应 Mask。

## 5. Embedding 是可学习查表

完成 Tokenizer 后，`"我爱"` 已经变成 `[1,2]`，但还不能把 1 和 2 当成能够比较含义的数值。如果交换词表编号：

```text
我 = 900
爱 = 7
```

文本含义不会改变。编号之间的加减、大小和距离都没有语言意义。

模型真正需要的是：为每个 token 准备若干可以独立调整的特征数，并让训练决定这些数怎样取值。因此引入一张可学习表。id 只负责选中其中一行，选中的那一行才进入神经网络计算。

假设词表大小为 $V$，向量宽度为 $C$：

$$
E\in\mathbb{R}^{V\times C}
$$

token id $i$ 的向量就是矩阵第 $i$ 行：

$$
x_i=E[i]
$$

PyTorch 实现：

```python
embedding = torch.nn.Embedding(vocab_size, n_embd)
idx = torch.tensor([[1, 2, 3], [3, 2, 1]])
x = embedding(idx)

print(idx.shape)  # [B,T] = [2,3]
print(x.shape)    # [B,T,C]
```

Embedding 输出不是预先写好的词义定义。训练开始时通常是随机小数，反向传播会根据预测任务逐渐调整每一行。

所以不要把两层含义混在一起：

```text
token id：查哪一行
embedding vector：这一行当前保存的可学习表示
```

## 6. Embedding 与 One-hot Linear 的等价关系

token id $i$ 也可表示为长度 $V$ 的 one-hot 行向量 $o_i$。则：

$$
o_iE=E[i]
$$

Embedding 查表与 one-hot 乘矩阵在数学上等价，但查表无需构造大多数元素为零的 one-hot 向量，计算和存储更高效。

这也说明 Embedding 本质上仍是可学习矩阵。

## 7. 位置向量补充顺序信息

先比较两个序列：

```text
猫 咬 狗
狗 咬 猫
```

它们包含完全相同的三个 token，只是顺序不同。仅查 token embedding 时，两段输入都只会取出“猫”“咬”“狗”对应的三行；同一个“猫”无论出现在第 0 位还是第 2 位，初始 token 向量都相同。

如果后续计算只知道 token 身份而没有位置，它没有足够信息区分谁先出现。为此必须把顺序信息另外加入每个位置的初始状态。

配套代码使用可学习绝对位置 Embedding：

```python
positions = torch.arange(T, device=idx.device)
token_vectors = self.token_embedding(idx)       # [B,T,C]
position_vectors = self.position_embedding(positions)  # [T,C]
x = token_vectors + position_vectors            # [B,T,C]
```

相加时 `[T,C]` 沿 Batch 维广播到 `[B,T,C]`。

数学表达：

$$
x_{b,t}=E_{\text{token}}[\text{idx}_{b,t}]+E_{\text{pos}}[t]
$$

同一个 token 在不同位置拥有相同 token 向量，但会加上不同位置向量。

这不是说“位置向量单独等于先后关系的全部含义”。它只为模型提供可用的位置信号，后面的 Attention 和 MLP 再学习怎样利用该信号。

## 8. 位置表示的主要方案

| 方案 | 核心方式 | 入门理解 |
|---|---|---|
| 可学习绝对位置 | 为每个位置保存一个向量 | 简单直观，配套代码采用 |
| 正弦位置编码 | 用不同频率的正弦余弦确定位置 | 无需学习固定位置表 |
| RoPE | 在 Q、K 上按位置旋转 | 让注意力自然带入相对位置信息 |

第一次实现选择可学习绝对位置，是为了让数据流最清楚。学习现代开源 LLM 时再重点研究 RoPE。

## 9. 输入长度与上下文窗口

配套代码中：

```python
self.position_embedding = nn.Embedding(block_size, n_embd)
```

因此可直接处理的位置编号只有 `0` 到 `block_size-1`。若输入更长，`forward()` 会报错；生成时则只保留最后 `block_size` 个 token：

```python
idx_context = idx[:, -self.config.block_size:]
```

`block_size` 是教学模型的上下文窗口。它增加时：

- 单个样本包含更多上下文；
- Attention 分数矩阵从 $T^2$ 增长；
- 激活内存与计算显著增加；
- 数据集每个可采样窗口的要求也更高。

## 10. Tokenizer 与模型的边界检查

常见错误包括：

```text
Tokenizer 产生 id 100，但 embedding 只有 100 行
→ 合法下标为 0 到 99，发生越界

加载模型却重新用另一份语料创建字符词表
→ 相同 id 表示不同字符，模型输出失去意义

只保存模型参数，没有保存 Tokenizer
→ 无法可靠还原训练时的文字与 id 映射
```

因此 Checkpoint 同时保存词表：

```python
checkpoint = {
    "model": model.state_dict(),
    "tokenizer": tokenizer.state_dict(),
    "config": asdict(model.config),
}
```

## 11. 本章代码实验

```python
import torch

text = "我爱猫，我爱鱼"
chars = sorted(set(text))
stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for char, i in stoi.items()}

ids = torch.tensor([[stoi[c] for c in "我爱猫"]])
B, T = ids.shape
C = 4

token_embedding = torch.nn.Embedding(len(chars), C)
position_embedding = torch.nn.Embedding(16, C)

token_vectors = token_embedding(ids)
position_vectors = position_embedding(torch.arange(T))
x = token_vectors + position_vectors

print("ids:", ids, ids.shape)
print("token vectors:", token_vectors.shape)
print("position vectors:", position_vectors.shape)
print("combined:", x.shape)
print("decoded:", "".join(itos[int(i)] for i in ids[0]))
```

把 `"我爱猫"` 改为 `"猫爱我"`，观察 token id 集合相似但位置与顺序发生变化。

## 12. 本章验收

- 能解释 Tokenizer 的 encode 与 decode。
- 能说明字符级、词级和子词级方案的主要取舍。
- 能根据 `Embedding(V,C)` 推断 `[B,T] → [B,T,C]`。
- 能解释 Embedding 与 one-hot 矩阵乘法的关系。
- 能说明 token embedding 与 position embedding 各自提供什么信息。
- 能解释 `[T,C]` 如何与 `[B,T,C]` 相加。
- 能说明模型与 Tokenizer 必须一起保存的原因。

详细 BPE 手算可在需要时查阅原版 `../LLM学习路线/03_Tokenizer_Attention_Transformer.md`。
