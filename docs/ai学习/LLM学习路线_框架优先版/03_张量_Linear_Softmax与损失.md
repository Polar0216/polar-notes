# 第二遍之一：张量、Linear、Softmax 与损失

本章拆解 GPT 全局图中的三处基础运算：隐藏状态如何经过 Linear 改变最后一维、logits 如何变成概率、模型如何用交叉熵衡量下一个 token 的预测质量。

## 1. 张量是带形状的数字容器

在 PyTorch 中，张量同时包含数据、形状、数据类型和设备：

```python
import torch

idx = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long)
print(idx.shape)   # torch.Size([2, 3])
print(idx.dtype)   # torch.int64
print(idx.device)  # cpu 或 cuda
```

这个例子中：

$$
B=2,\qquad T=3
$$

因此 `idx.shape == [B,T]`。

模型常见张量如下：

| 名称 | 形状 | 数据含义 | 常见类型 |
|---|---|---|---|
| token ids | `[B,T]` | 每个位置的词表编号 | `torch.long` |
| hidden state | `[B,T,C]` | 每个位置的连续向量 | 浮点数 |
| attention scores | `[B,H,T,T]` | 每个头中位置两两之间的分数 | 浮点数 |
| logits | `[B,T,V]` | 每个位置对整个词表的分数 | 浮点数 |
| targets | `[B,T]` | 正确的下一个 token 编号 | `torch.long` |
| loss | 标量 | 整批预测错误的平均值 | 浮点数 |

不要先背整张表。先用具体数组读懂 `[B,T,C]`。

假设：

```text
B = 2：有两条文本
T = 3：每条文本取三个 token
C = 4：每个 token 用四个数表示
```

于是 `x.shape == [2,3,4]`。它不是一个无法展开的“三维整体”，而是：

```text
第 0 条文本
  第 0 个 token → [四个数]
  第 1 个 token → [四个数]
  第 2 个 token → [四个数]

第 1 条文本
  第 0 个 token → [四个数]
  第 1 个 token → [四个数]
  第 2 个 token → [四个数]
```

用下标读取：

```python
x[b, t, c]  # 第 b 条文本、第 t 个 token、第 c 个特征
x[b, t, :]  # 固定文本和 token，取它的完整 C 维向量
x[b, :, :]  # 固定一条文本，取它的全部 token 向量
```

这给出一条稳定的读法：

> 形状中的每个位置对应一个下标；固定其他下标，只让某个下标变化，就沿着那个轴取得一组数据。

因此 $T$ 不是“第二个神秘维度”，而是 token 位置下标可以取多少个值；$C$ 是每个 token 内部的特征下标可以取多少个值。

其他形状可以按同样方式读取：

```text
q[b,h,t,:]         第 b 条文本、第 h 个头、第 t 个 token 的 D 维 Query
attention[b,h,t,:] 第 b 条文本、第 h 个头中，位置 t 对全部 T 个位置的权重
logits[b,t,:]      第 b 条文本、位置 t 对全部 V 个词表候选的分数
targets[b,t]       与 logits[b,t,:] 对应的唯一正确 token id
```

`loss` 最终对许多位置取平均，因此是一个标量，`loss.shape == torch.Size([])`。

`shape` 不是附属信息。神经网络中的大多数报错都可以先翻译为：哪个下标代表什么，以及哪两个维度不兼容。

## 2. 矩阵乘法与最后一维变换

GPT 内部经常需要保留“这是第几条文本、这是第几个 token”，只改变“这个 token 用多少个特征表示”。Linear 正是对每个 token 的最后一维执行同一个变换。

对二维矩阵：

$$
X\in\mathbb{R}^{B\times C_{\text{in}}},
\qquad
W\in\mathbb{R}^{C_{\text{in}}\times C_{\text{out}}}
$$

矩阵乘法得到：

$$
Z=XW
\in\mathbb{R}^{B\times C_{\text{out}}}
$$

内侧维度 $C_{\text{in}}$ 必须相等，结果保留外侧维度。

PyTorch 的 `nn.Linear(in_features, out_features)` 在语义上对输入的最后一维执行同一种变换。若：

```text
x.shape = [B,T,C_in]
```

则：

```python
layer = torch.nn.Linear(C_in, C_out)
z = layer(x)
```

结果为：

```text
z.shape = [B,T,C_out]
```

Batch 维 $B$ 和序列维 $T$ 不参与权重维度匹配，相同的 Linear 参数被应用到每个样本、每个位置。

把具体形状代入会更清楚。设：

```text
x.shape = [2,3,4]
W.shape = [4,5]
```

`x` 中共有 $2\times3=6$ 个 token 向量，每个向量长度为 4：

```text
x[0,0,:]  [4]
x[0,1,:]  [4]
x[0,2,:]  [4]
x[1,0,:]  [4]
x[1,1,:]  [4]
x[1,2,:]  [4]
```

每一个都乘同一个 `[4,5]` 矩阵：

$$
[4]\times[4,5]\longrightarrow[5]
$$

所以六个长度为 4 的向量分别变成六个长度为 5 的向量：

$$
[2,3,4]\times[4,5]\longrightarrow[2,3,5]
$$

也可以暂时把前两维合并：

$$
[B,T,C]
\longrightarrow[B\!\cdot\!T,C]
\longrightarrow[B\!\cdot\!T,V]
\longrightarrow[B,T,V]
$$

这里被求和消去的是匹配的 $C$，$B$ 和 $T$ 只是说明有多少个长度为 $C$ 的向量，不参与这一层的特征点积。

用一个真正的向量计算：

$$
x=[1,2,3],
\qquad
W=
\begin{bmatrix}
1&2\\
3&4\\
5&6
\end{bmatrix}
$$

则：

$$
xW=
\left[
1\times1+2\times3+3\times5,
\quad
1\times2+2\times4+3\times6
\right]
=[22,28]
$$

这就是 `[C] × [C,V] → [V]`；GPT 的 `[B,T,C] × [C,V] → [B,T,V]` 只是同时处理了许多 token 向量。

## 3. Linear 层的行向量约定

教程统一使用“样本放在行中”的约定：

$$
Z=XW+b
$$

其中：

$$
X\in\mathbb{R}^{N\times D_{\text{in}}},\quad
W\in\mathbb{R}^{D_{\text{in}}\times D_{\text{out}}},\quad
b\in\mathbb{R}^{D_{\text{out}}}
$$

因此：

$$
Z\in\mathbb{R}^{N\times D_{\text{out}}}
$$

PyTorch 内部保存的 `layer.weight.shape` 是 `[D_out,D_in]`，前向传播等价于：

$$
Z=XW_{\text{PyTorch}}^{\mathsf T}+b
$$

这只是参数存储约定不同，不是数学矛盾。

```python
layer = torch.nn.Linear(3, 5)
print(layer.weight.shape)  # [5, 3]
print(layer.bias.shape)    # [5]

x = torch.randn(2, 4, 3)
z = layer(x)
print(z.shape)             # [2, 4, 5]
```

## 4. 广播使偏置作用于每个位置

假设：

```text
XW.shape = [B,T,C_out]
b.shape  = [C_out]
```

加法从右向左比较形状，`b` 可视为 `[1,1,C_out]`，沿 $B$ 和 $T$ 复用：

$$
[B,T,C_{\text{out}}]+[C_{\text{out}}]
\longrightarrow
[B,T,C_{\text{out}}]
$$

广播不复制实际存储，也不把不同样本混在一起。它表示每个位置都加上同一个可学习偏置向量。

广播判断规则：从最右侧对齐，每对维度必须相等，或其中一个为 1，或较短形状在该处不存在。

## 5. Linear 在 GPT 中的重复出现

同一种最后一维变换承担多种职责：

| 位置 | 形状变化 | 作用 |
|---|---|---|
| QKV 投影 | `[B,T,C] → [B,T,3C]` | 为 Attention 产生 Query、Key、Value |
| Attention 输出投影 | `[B,T,C] → [B,T,C]` | 混合各注意力头结果 |
| MLP 第一层 | `[B,T,C] → [B,T,4C]` | 扩大特征空间 |
| MLP 第二层 | `[B,T,4C] → [B,T,C]` | 投影回残差通道 |
| LM Head | `[B,T,C] → [B,T,V]` | 对词表中的每个 token 打分 |

掌握 Linear 不是为了只写一个全连接分类器，而是为了理解 GPT 中大量投影层。

## 6. Logits 是未归一化分数

LM Head 输出：

$$
Z\in\mathbb{R}^{B\times T\times V}
$$

对某个样本的某个位置，可能得到：

```text
logits = [2.0, 1.0, -1.0]
```

这些值：

- 可以为负数；
- 总和不必为 1；
- 只表达候选 token 的相对偏好；
- 平移同一个常数不会改变 Softmax 概率。

最后一点可由公式看出：

$$
\frac{e^{z_i+c}}{\sum_j e^{z_j+c}}
=
\frac{e^c e^{z_i}}{e^c\sum_j e^{z_j}}
=
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

## 7. Softmax 沿词表维产生概率

Softmax 定义为：

$$
p_i=\frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}
$$

它保证：

$$
p_i>0,
\qquad
\sum_{i=1}^{V}p_i=1
$$

数值稳定实现会先减去最大值：

```python
def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    exponentials = torch.exp(shifted)
    return exponentials / exponentials.sum(dim=dim, keepdim=True)
```

减去最大值不改变结果，却保证最大指数为 $e^0=1$，减少指数溢出。

在 GPT 的 `[B,T,V]` logits 上：

```python
probs = torch.softmax(logits, dim=-1)
```

`dim=-1` 指 $V$，也就是每个位置的全部候选 token。但“沿最后一维计算”很容易被误读成“对最后一列计算”，两者不是一回事。

先看二维数组：

```text
x.shape = [2,3]

x = [[1,2,3],
     [4,5,6]]
```

两个下标是 `x[row,column]`：

```text
dim=0：让 row 下标变化，分别取得 [1,4]、[2,5]、[3,6]
dim=1：让 column 下标变化，分别取得 [1,2,3]、[4,5,6]
dim=-1：最后一个维度，二维时等于 dim=1
```

因此：

```python
torch.softmax(x, dim=-1)
```

会对每一行内部的三个元素分别做 Softmax，并使每一行的和为 1。它之所以可口语化为“横着算”，是因为变化的是列下标；正式理解仍应是“最后一个下标变化”。

回到三维 logits：

```text
logits[b,t,v]
```

执行 `dim=-1` 时固定 $b$ 和 $t$，只让 $v$ 从 0 变化到 $V-1$：

```text
logits[b,t,:]
→ 当前样本、当前位置的全部词表候选
→ 单独做一次 Softmax
```

结果仍是 `[B,T,V]`，但每个固定 $(b,t)$ 对应的 $V$ 个概率之和为 1。

可以直接验证两个方向：

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

along_last = torch.softmax(x, dim=-1)
along_first = torch.softmax(x, dim=0)

print(along_last.sum(dim=-1))   # 两个 1：每行分别归一化
print(along_first.sum(dim=0))   # 三个 1：每列分别归一化
```

## 8. 交叉熵衡量正确 token 的概率

若真实 token 编号为 $y$，单位置交叉熵为：

$$
L=-\log p_y
$$

它不仅判断预测类别是否正确，还区分置信程度：

| 正确 token 概率 | 损失 $-\log p_y$ | 含义 |
|---:|---:|---|
| 0.90 | 0.105 | 高置信度正确 |
| 0.50 | 0.693 | 不确定 |
| 0.10 | 2.303 | 给正确答案的概率很低 |
| 0.01 | 4.605 | 高置信度错误 |

PyTorch 正确用法：

```python
loss = torch.nn.functional.cross_entropy(logits_2d, targets_1d)
```

不要先写：

```python
probs = torch.softmax(logits_2d, dim=-1)
loss = torch.nn.functional.cross_entropy(probs, targets_1d)  # 错误语义
```

`cross_entropy` 需要原始 logits，内部会以稳定方式结合 LogSoftmax 与负对数似然。

## 9. 序列损失的形状整理

完整模型中：

```text
logits  [B,T,V]
targets [B,T]
```

展平后：

```text
logits.reshape(B*T,V)  [B*T,V]
targets.reshape(B*T)   [B*T]
```

代码：

```python
B, T, V = logits.shape
loss = F.cross_entropy(
    logits.reshape(B * T, V),
    targets.reshape(B * T),
)
```

每一行 logits 与同位置 target 仍一一对应。损失默认对全部 $B\times T$ 个位置取平均。

## 10. 本章代码实验

将以下内容保存到临时 Python 文件或在交互环境执行：

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, C, V = 2, 3, 4, 6
x = torch.randn(B, T, C)
lm_head = torch.nn.Linear(C, V)
logits = lm_head(x)
targets = torch.randint(0, V, (B, T))

print("x:", x.shape)
print("weight:", lm_head.weight.shape)
print("logits:", logits.shape)
print("targets:", targets.shape)

probs = torch.softmax(logits, dim=-1)
print("probability sums:", probs.sum(dim=-1))

loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
print("loss:", loss.item())
```

预期结果：`probability sums` 中每个值接近 1，`loss` 是一个标量。

## 11. 本章验收

- 能根据 `nn.Linear(64,100)` 推断 `[8,32,64] → [8,32,100]`。
- 能解释 PyTorch Linear 权重形状为何是 `[out_features,in_features]`。
- 能说明偏置 `[C]` 如何加到 `[B,T,C]`。
- 能区分 logits 与概率。
- 能说明 GPT 中 Softmax 的 `dim=-1` 对应词表维。
- 能把 `[B,T,V]` 与 `[B,T]` 整理为交叉熵需要的形状。
- 能解释训练时不应在 `cross_entropy` 前手动 Softmax。

更细的 NumPy 推导可查原版 `../LLM学习路线/02_NumPy神经网络与PyTorch.md`，但先完成本章实验，再按问题查阅。
