# 第二遍之三：Attention 与 Transformer Block

本章开始展开 Transformer Block 的内部计算。输入和输出形状都保持 `[B,T,C]`，其中依次完成位置之间的信息交换、逐位置特征加工、残差连接与归一化。

## 1. 初始 token 向量需要吸收上下文

Embedding 查表时只使用 token id，因此同一个 token 总是取得同一行初始向量。比较：

```text
苹果很甜。
苹果发布了新产品。
```

两处“苹果”的初始 token embedding 相同，但经过上下文处理后，我们希望它们的隐藏状态分别包含“水果”和“公司”相关信息。

这给 Transformer 提出一个明确任务：

```text
输入：只含 token 身份与位置的初始向量
输出：吸收当前上下文后得到的更新向量
```

把所有词平均起来虽然能混入上下文，却无法根据当前位置动态区分哪些内容重要。Attention 的核心改进是：先计算当前位置与每个可见位置的相关程度，再按相关程度汇总信息。

这正是 3Blue1Brown 可视化中“初始 embedding 逐层变成 context-rich embedding”的主线。可视化帮助我们理解想要发生的变化；真实参数究竟编码了哪些语义，仍由训练结果决定。

## 2. Transformer Block 的整体结构

配套代码采用 Pre-Norm Block：

```python
def forward(self, x):
    x = x + self.attention(self.norm_before_attention(x))
    x = x + self.mlp(self.norm_before_mlp(x))
    return x
```

数据流为：

```text
x [B,T,C]
│
├─────────────── residual ───────────────┐
│                                        │
└→ LayerNorm → Causal Self-Attention ────┴→ x [B,T,C]
                                          │
                 ┌──── residual ──────────┤
                 │                        │
                 └→ LayerNorm → MLP ─────┴→ output [B,T,C]
```

四个角色：

| 组件 | 核心职责 |
|---|---|
| Attention | 让不同 token 位置按内容交换信息 |
| MLP | 在每个位置内部重新组合与扩展特征 |
| Residual | 保留原信息，并为梯度提供较短路径 |
| LayerNorm | 调整每个位置内部特征的尺度与分布 |

## 3. Query、Key、Value 分离匹配与传递

Attention 需要完成两个不同计算：

```text
匹配：当前位置应从哪些位置读取信息
传递：被选中的位置实际提供什么内容
```

如果所有位置直接共享原始 $X$ 做两件事，模型很难分别学习“用于判断相关的特征”和“真正传递的特征”。因此为同一个位置计算三种可学习表示。

输入隐藏状态：

$$
X\in\mathbb{R}^{B\times T\times C}
$$

每个位置通过三个可学习投影产生：

$$
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V
$$

三者在 self-attention 中都来自同一个 $X$，因此称为自注意力。

可以用下面的角色区分检查理解，但不要把它当成参数的字面语义：

```text
Query：当前位置拿什么特征去匹配
Key：候选位置拿什么特征接受匹配
Value：候选位置被赋予权重后实际传递的内容
```

Query 与 Key 点积只决定权重，最终被加权求和的是 Value。这正是第三种投影不能简单省略的原因。

配套代码为了效率用一个 Linear 同时产生三者：

```python
qkv = self.qkv(x)                    # [B,T,3C]
query, key, value = qkv.split(C, dim=-1)
```

这等价于三个独立投影按最后一维拼接，但实现更紧凑。

## 4. 单头注意力的信息交换

暂时忽略 Batch 和多头，设：

$$
Q,K,V\in\mathbb{R}^{T\times D}
$$

注意力分数：

$$
S=\frac{QK^{\mathsf T}}{\sqrt{D}}
\in\mathbb{R}^{T\times T}
$$

$S_{i,j}$ 表示位置 $i$ 的 Query 与位置 $j$ 的 Key 的匹配程度。每一行对应“当前位置从所有可见位置读取信息的偏好”。

因果遮罩后执行 Softmax：

$$
A=\operatorname{softmax}(S+M)
$$

其中不可见位置的 Mask 值为 $-\infty$。Softmax 后这些位置的权重为 0。

最后对 Value 加权求和：

$$
O=AV
\in\mathbb{R}^{T\times D}
$$

整条计算可压缩为：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(
\frac{QK^{\mathsf T}}{\sqrt D}+M
\right)V
$$

## 5. 缩放因子保持 Softmax 稳定

如果暂时不缩放，$D$ 增大意味着点积中相加的项更多。即使各维数值范围相近，$QK^{\mathsf T}$ 的典型幅度也会随 $D$ 增长。大幅度分数进入 Softmax 后，最大项可能接近 1，其余项接近 0，训练梯度变得过于集中。

除以 $\sqrt D$ 可让分数尺度在不同 head dimension 下更稳定：

$$
S=\frac{QK^{\mathsf T}}{\sqrt D}
$$

它不是为了改变形状，而是为了控制数值尺度。

## 6. 因果遮罩限制未来信息

长度 $T=4$ 时，下三角 Mask 为：

$$
M=
\begin{bmatrix}
0 & -\infty & -\infty & -\infty\\
0 & 0 & -\infty & -\infty\\
0 & 0 & 0 & -\infty\\
0 & 0 & 0 & 0
\end{bmatrix}
$$

配套代码：

```python
mask = torch.tril(torch.ones(block_size, block_size))
scores = scores.masked_fill(
    mask[:, :, :T, :T] == 0,
    float("-inf"),
)
weights = torch.softmax(scores, dim=-1)
```

Softmax 沿每行最后一维执行，因为每个 Query 位置需要在全部 Key 位置中分配读取权重。

Causal Mask 与 Padding Mask 作用不同：

- Causal Mask 遮住未来位置，保证自回归训练合法。
- Padding Mask 遮住为了批量补齐而添加的无意义位置。
- 对变长批量训练，两种 Mask 可能同时存在。

## 7. 多头注意力的形状变换

设：

$$
C=H\times D
$$

即总隐藏宽度分成 $H$ 个头，每个头宽度为 $D$。

形状变化：

```text
Q、K、V
[B,T,C]
→ view
[B,T,H,D]
→ transpose
[B,H,T,D]
```

每个头独立计算：

```text
scores  [B,H,T,T]
weights [B,H,T,T]
context [B,H,T,D]
```

再合并：

```text
[B,H,T,D]
→ transpose
[B,T,H,D]
→ contiguous + view
[B,T,C]
```

配套代码中的核心部分：

```python
query = query.view(B, T, H, D).transpose(1, 2)
key = key.view(B, T, H, D).transpose(1, 2)
value = value.view(B, T, H, D).transpose(1, 2)

scores = query @ key.transpose(-2, -1)  # [B,H,T,T]
context = weights @ value               # [B,H,T,D]
context = context.transpose(1, 2).contiguous().view(B, T, C)
```

`n_embd` 必须能被 `n_head` 整除，否则无法让各头拥有相同的 $D$。

## 8. 多头机制提供多个表示子空间

单头只有一套 QKV 投影和一张注意力图。多头让模型在多个子空间中并行建立不同关系，某些头可能更关注近邻、标点、实体指代或句法模式。

这种解释是可能出现的行为，不应把某个头固定拟人化为唯一功能。训练只优化最终预测损失，不会给每个头预先指定语言学职责。

各头结果拼回 `[B,T,C]` 后，还会经过输出投影：

$$
Y=\operatorname{Concat}(O_1,\ldots,O_H)W_O
$$

它用于混合各头产生的信息。

## 9. MLP 完成逐位置特征加工

Attention 结束后，当前位置得到的是若干 Value 的加权组合。加权汇总擅长把其他位置的信息带回来，但仅靠线性加权不能完成所有条件化的非线性特征变换。

因此每个位置还要经过同一个 MLP，把已经取得的信息与当前位置原有特征进一步组合。MLP 不再读取其他位置，同一组权重独立作用于所有 token：

$$
\operatorname{MLP}(x)
=
W_2\operatorname{GELU}(W_1x+b_1)+b_2
$$

典型形状：

$$
[B,T,C]
\longrightarrow
[B,T,4C]
\longrightarrow
[B,T,C]
$$

配套实现：

```python
self.network = nn.Sequential(
    nn.Linear(C, 4 * C),
    nn.GELU(),
    nn.Linear(4 * C, C),
    nn.Dropout(dropout),
)
```

可以用一句没有额外拟人化的描述区分二者：

```text
Attention 沿 T 维混合不同位置的信息。
MLP 沿 C 维非线性变换单个位置的特征。
```

扩大到 $4C$ 提供更多中间特征方向，再投影回 $C$ 以满足残差相加。倍率 4 是常见设计选择，不是逻辑定律。

## 10. 残差连接保留信息与梯度路径

若直接令新状态等于子层输出：

$$
x_{\text{new}}=f(x)
$$

子层必须同时完成两件事：保留旧状态中的有用内容，并加入本层的新计算。层数增加后，每一层都完整重建表示会提高优化难度。

Residual 将子层定义为一个增量：

$$
y=x+f(x)
$$

现在 $f(x)$ 只需学习当前层应补充的修正。反向传播时：

$$
\frac{\partial y}{\partial x}
=
I+\frac{\partial f}{\partial x}
$$

恒等路径 $I$ 为深层网络提供直接梯度通道，降低所有梯度都必须穿过复杂子层的困难。

由于要执行加法，`x` 与 `f(x)` 形状必须一致。这正是 Attention 和 MLP 最终都返回 `[B,T,C]` 的原因。

## 11. LayerNorm 稳定每个位置的特征

隐藏状态经过很多层矩阵乘法、激活、Attention 和残差相加后，不同层收到的数值尺度可能不断变化，例如某些特征绝对值逐渐变大，另一些接近零。后续子层面对持续漂移的输入尺度时，参数优化会更困难。

LayerNorm 在每个 token 内部，沿 $C$ 个特征计算均值与方差，使进入重要子层的数值尺度更容易控制。它不是把所有 token 混在一起，也不会改变张量形状。

对某个位置的 $C$ 个特征，LayerNorm 计算：

$$
\mu=\frac{1}{C}\sum_{i=1}^{C}x_i,
\qquad
\sigma^2=\frac{1}{C}\sum_{i=1}^{C}(x_i-\mu)^2
$$

归一化并加入可学习缩放、平移：

$$
y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i
$$

它不改变形状，只调整数值分布。`nn.LayerNorm(C)` 默认对输入最后一维进行归一化，因此适合 `[B,T,C]`。

Pre-Norm 表示先归一化再进入子层：

```python
x = x + attention(layer_norm(x))
```

## 12. 两个 token 的最小手算

设单头且 $D=2$：

$$
Q=
\begin{bmatrix}
1&0\\
1&1
\end{bmatrix},
\quad
K=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
\quad
V=
\begin{bmatrix}
2&0\\
0&4
\end{bmatrix}
$$

未缩放分数：

$$
QK^{\mathsf T}=
\begin{bmatrix}
1&0\\
1&1
\end{bmatrix}
$$

加入因果遮罩后，第一行只能读取第一个 token，因此第一行权重为 `[1,0]`，输出为 `[2,0]`。

第二行可以读取两个 token，两个分数相同，Softmax 权重为 `[0.5,0.5]`，输出为：

$$
0.5[2,0]+0.5[0,4]=[1,2]
$$

这展示了 Attention 的核心：输出是可见 Value 的加权组合。

## 13. 本章代码观察任务

打开 `code/mini_gpt_complete.py` 并完成：

1. 在 `CausalSelfAttention.forward()` 中打印 Q、scores、weights、context 的形状。
2. 将 `n_embd=64,n_head=4`，写出 $D=16$ 并核对输出。
3. 临时打印第一批第一个头的注意力权重，确认右上角为 0。
4. 删除 Causal Mask 仅观察矩阵变化，随后立即恢复；说明该模型为何会发生未来泄漏。
5. 将 Transformer 层数从 1 改为 3，确认每个 Block 前后形状仍为 `[B,T,C]`。

打印数值时使用极小的 $B$、$T$，避免输出无法阅读。

## 14. 本章验收

- 能写出缩放点积注意力公式并说明每个矩阵的语义。
- 能推导 `[B,T,C] → [B,H,T,D] → [B,H,T,T] → [B,T,C]`。
- 能解释 Softmax 在 Attention 中沿 Key 位置维执行。
- 能区分 Causal Mask 与 Padding Mask。
- 能说明 Attention 与 MLP 的职责差异。
- 能说明残差相加要求两侧形状一致。
- 能说明 LayerNorm 不改变形状，并在最后一维归一化。

需要更多纸笔推导时，查阅原版 `../LLM学习路线/03A_Attention与Transformer手算.md`。
