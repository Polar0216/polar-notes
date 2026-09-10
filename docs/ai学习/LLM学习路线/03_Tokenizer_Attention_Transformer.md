# 模块 3：Tokenizer、Attention 与 Transformer

## 目标

把“文本→token id→向量→上下文表示→下一个 token 概率”整条链路打通。完成字符、词和 BPE tokenizer，实现可视化的因果自注意力与一个 Transformer Block。

> Attention 的每一步矩阵运算、Mask、LayerNorm 和残差推导见 [Attention 与 Transformer 手算讲义](03A_Attention与Transformer手算.md)。建议在写代码前先完整算一遍两 token 示例。

## 学习导航

按“正文概念 → 补充讲义手算 → 文内小例子与测试 → 验收与复盘 → 文末 Project”的顺序学习。补充讲义用于推导，不替代主模块的实现练习。

## 1. 语言模型到底预测什么

给定 token 序列 `x1...xt`，自回归语言模型学习：

$
P(x_1,...,x_T)=\prod_{t=1}^{T}P(x_t\mid x_{<t})
$

它不是一次“写完一句话”，而是反复预测下一个 token，再把新 token 接到输入后继续预测。

## 2. Tokenizer 是模型与文字之间的合同

模型只接收整数。Tokenizer 负责：

```text
原始字符串 → 规范化/预分词 → token 序列 → id 序列
id 序列 → token 序列 → 字符串
```

Tokenizer 与模型权重必须匹配。换 tokenizer 会改变 id 的含义和词表大小，原来的 embedding 行不再对应原 token。

### 字符级

每个字符是 token。实现简单、几乎无未知字符，但序列很长；“学”“习”分别建模。

```python
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return [stoi[ch] for ch in s]

def decode(ids):
    return "".join(itos[i] for i in ids)
```

### 词级

英文可按空格和标点切分，但中文没有天然空格；词表会爆炸，新词变成 `<unk>`，并且邮箱、代码、数字很难穷举。

### 子词级

BPE、Unigram 等在字符与整词之间折中：常见片段合并成一个 token，罕见词拆成多个片段。主流 LLM 多采用某种子词/字节方案。

## 3. 特殊 token

常见特殊符号：

- `<bos>`：序列开始。
- `<eos>`：序列结束；生成停止的重要信号。
- `<pad>`：batch 内补齐长度，通常应在损失/注意力里屏蔽。
- `<unk>`：词表外项，字节级方案可能避免它。
- 对话模板 token：标识 system/user/assistant 边界。

不要假设 `pad_id=0` 或 `eos_id=2`；从 tokenizer 配置读取。对话模型使用错误 chat template 会显著降低表现。

## 4. BPE 手工推演

训练语料先表示成最小单元。每轮统计相邻 token 对，合并频率最高的一对，直到达到词表大小或次数阈值。

伪代码：

```python
def train_bpe(corpus_tokens, target_vocab_size):
    vocab = initial_symbols(corpus_tokens)
    merges = []
    while len(vocab) < target_vocab_size:
        pair_counts = count_adjacent_pairs(corpus_tokens)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        new_symbol = best_pair[0] + best_pair[1]
        corpus_tokens = merge_everywhere(corpus_tokens, best_pair, new_symbol)
        merges.append(best_pair)
        vocab.add(new_symbol)
    return vocab, merges
```

编码新文本时按训练得到的 merge 顺序合并，而不是重新按当前文本频率训练。你还需明确：规范化、空格如何保留、Unicode/字节如何处理、特殊 token 是否参与合并。

### Tokenizer 评价

- 可逆性：`decode(encode(text))` 是否符合预期。
- 覆盖率：中英混合、emoji、数字、代码、罕见字符是否可处理。
- 压缩率：每字符/每字节需要多少 token。
- 词表大小与序列长度的权衡。
- 边界稳定性：前导空格、换行是否改变 token。

## 5. Embedding：id 变向量

Embedding 是一个可学习查找表 `E:[V,C]`。id 为 `k` 就取第 `k` 行。它等价于 one-hot 向量乘矩阵，但无需真的构造巨大的 one-hot。

Token embedding 只说明“是什么 token”，还需要位置信息说明“在第几个位置”。原始 Transformer 使用正弦位置编码；GPT 可用可学习绝对位置，现代模型也常用 RoPE。入门先实现可学习位置 embedding：

```python
tok = token_embedding(idx)                 # [B,T,C]
pos = position_embedding(torch.arange(T))  # [T,C]
x = tok + pos                              # 广播到 [B,T,C]
```

## 6. Attention 的直觉

一句话里的每个位置都提出一个问题（Query），暴露自己可被匹配的标签（Key），并携带可聚合的信息（Value）。

$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$

$
A=softmax(\frac{QK^T}{\sqrt{d_k}}+M),\quad O=AV
$

形状（单头）：

```text
X       [B,T,C]
Wq      [C,D]
Q/K/V   [B,T,D]
K^T     [B,D,T]
scores  [B,T,T]
weights [B,T,T]
output  [B,T,D]
```

`scores[b,i,j]` 表示位置 i 对位置 j 的关注分数。除以 `sqrt(D)` 是为了避免维度大时点积方差过大，Softmax 过早饱和导致梯度很小。

## 7. 因果 Mask

训练“预测下一个 token”时，位置 i 不能偷看未来位置。上三角未来区域加上负无穷，Softmax 后概率变成 0：

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(mask.to(scores.device), float("-inf"))
weights = torch.softmax(scores, dim=-1)
```

Padding mask 与 causal mask 不是一回事：前者忽略补齐位置，后者阻止看未来。构造 mask 后要测试每行概率和为 1、未来位置权重接近 0。

## 8. 多头注意力

多个头让模型在不同子空间学习不同关系。设 `C = H * D`：

```text
Q [B,T,C]
reshape → [B,T,H,D]
transpose → [B,H,T,D]
attention → [B,H,T,D]
transpose/contiguous/reshape → [B,T,C]
output projection → [B,T,C]
```

`transpose` 后内存可能不连续，`view` 可能失败或误用；常使用 `transpose(...).contiguous().view(...)` 或 `reshape`，同时仍应理解内存布局。

## 9. Transformer Block

典型 pre-norm GPT block：

$
x=x+Attention(LayerNorm(x))
$

$
x=x+MLP(LayerNorm(x))
$

残差连接提供信息和梯度的“高速公路”；LayerNorm 稳定每个 token 特征维的尺度；MLP 对每个位置独立做非线性变换；Attention 在位置之间交换信息。

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

## 10. Attention 可视化应如何解释

热力图横轴 key、纵轴 query。因果模型上三角应被遮住。注意力权重高不等于“这是模型做决定的唯一原因”；残差、MLP、多个层头都会参与，注意力图更适合调试和形成假设，不应当作完整因果解释。

建议用人造任务验证：例如序列 `A x x A`，训练模型预测相同符号或括号匹配，看某些头是否学会关注对应位置。

## 验收与复盘

- Tokenizer round-trip 与保存加载一致。
- Attention 未来位置权重为 0。
- 权重最后一维和为 1。
- 全部张量无 NaN/Inf。
- 不同 batch 样本互不泄漏。
- 固定 seed、关闭 dropout 时，相同输入输出相同。

### 达标条件

你能在白纸上写出 Q/K/V 形状和公式；解释缩放、mask、多头、残差与 LayerNorm；从零实现 BPE 和 Transformer Block；看 `nanoGPT` 的 attention 代码时能沿着形状读下去。

## 课后 Project

### Project 3：Tokenizer

依次实现字符级、简单词级和 BPE，提供 `train_tokenizer.py`、`encode.py`、`decode.py` 和序列化词表。

验收数据至少覆盖：中文、英文、空格换行、emoji、代码、空字符串、未知字符。比较三者的词表大小、平均 token 数和可逆性。

### Project 4：Attention 可视化

只用 PyTorch 基础算子实现单头和多头 causal attention，不能直接调用 `nn.MultiheadAttention`。画出 mask 前分数、mask 后权重和输出。

### Project 4.5：Transformer Block

组合 LayerNorm、MHA、MLP 和残差；输入 `[2,16,64]`，输出必须同形状。检查反向传播后每个参数有有限梯度。
