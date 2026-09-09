# Attention 与 Transformer 手算讲义

本讲义把模块 3 中最容易“看懂代码、没懂原理”的部分展开。建议准备纸笔，逐项验证矩阵形状。

## 讲义定位与使用方式

这是模块 3 的推导补充。先读主模块建立 Tokenizer 与 Transformer 全局链路，再在这里逐行手算 Attention；完成文末纸笔练习后回到主模块实现和 Project。

## 1. 为什么普通 MLP 不适合直接处理序列关系

假设一句话有 $T$ 个 token，每个 token 的表示维度为 $C$：

$$
X\in\mathbb{R}^{T\times C}
$$

逐位置 MLP 对每个 token 独立执行同一函数：

$$
H_t=f(X_t)
$$

它能改变单个 token 的特征，却不能让第 $t$ 个 token 读取第 $s$ 个 token 的信息。卷积可以在固定窗口交换信息，但若想连接很远的位置，需要堆叠很多层。

Self-Attention 的目标是让每个位置根据当前内容，动态选择应该从哪些位置读取信息。

## 2. Query、Key、Value 的类比及边界

可以把注意力想象成检索：

- Query：当前位置想找什么；
- Key：每个位置提供什么匹配标签；
- Value：若匹配成功，真正取走的信息。

但它们不是人工编写的自然语言问题、键和值，而是由训练学习出的向量：

$$
Q=XW_Q,\qquad
K=XW_K,\qquad
V=XW_V
$$

其中：

$$
W_Q,W_K,W_V
\in
\mathbb{R}^{C\times D}
$$

所以：

$$
Q,K,V
\in
\mathbb{R}^{T\times D}
$$

## 3. 两个 token 的完整手算

为方便计算，暂时忽略 batch 维。设两个 token 的输入表示：

$$
X=
\begin{bmatrix}
1&0\\
1&1
\end{bmatrix}
$$

选择非常简单的投影：

$$
W_Q=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
\qquad
W_K=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
$$

$$
W_V=
\begin{bmatrix}
1&2\\
0&1
\end{bmatrix}
$$

于是：

$$
Q=XW_Q=
\begin{bmatrix}
1&0\\
1&1
\end{bmatrix}
$$

$$
K=XW_K=
\begin{bmatrix}
1&0\\
1&1
\end{bmatrix}
$$

$$
V=XW_V=
\begin{bmatrix}
1&2\\
1&3
\end{bmatrix}
$$

### 3.1 计算相似度分数

每个 Query 与每个 Key 做点积：

$$
S=QK^\mathsf{T}
$$

$$
K^\mathsf{T}
=
\begin{bmatrix}
1&1\\
0&1
\end{bmatrix}
$$

因此：

$$
S=
\begin{bmatrix}
1&0\\
1&1
\end{bmatrix}
\begin{bmatrix}
1&1\\
0&1
\end{bmatrix}
=
\begin{bmatrix}
1&1\\
1&2
\end{bmatrix}
$$

解释每个元素：

$$
S_{i,j}=Q_i\cdot K_j
$$

$S_{2,1}=1$ 表示第 2 个位置的 Query 与第 1 个位置的 Key 的匹配分数为 $1$。

### 3.2 为什么除以 $\sqrt{D}$

如果 Query 和 Key 的各维独立、均值约为 $0$、方差约为 $1$，点积：

$$
Q\cdot K
=
\sum_{d=1}^{D}Q_dK_d
$$

的方差会随 $D$ 增大。维度越大，分数绝对值越容易很大，Softmax 变得极端饱和。

除以 $\sqrt{D}$ 后，分数尺度在不同维度下更稳定：

$$
\widetilde S
=
\frac{QK^\mathsf{T}}{\sqrt D}
$$

本例 $D=2$：

$$
\widetilde S
\approx
\begin{bmatrix}
0.707&0.707\\
0.707&1.414
\end{bmatrix}
$$

### 3.3 加入因果 Mask

自回归模型中，第 1 个位置不能读取第 2 个位置，因为第 2 个 token 对它而言属于未来。

Mask 可以写成：

$$
M=
\begin{bmatrix}
0&-\infty\\
0&0
\end{bmatrix}
$$

加入 Mask：

$$
\widetilde S+M
=
\begin{bmatrix}
0.707&-\infty\\
0.707&1.414
\end{bmatrix}
$$

### 3.4 对每一行做 Softmax

注意力权重：

$$
A=
\operatorname{softmax}
\left(
\widetilde S+M
\right)
$$

第一行只有第一个位置可见：

$$
A_1=
\begin{bmatrix}
1&0
\end{bmatrix}
$$

第二行近似为：

$$
A_2
=
\operatorname{softmax}
\left(
\begin{bmatrix}
0.707&1.414
\end{bmatrix}
\right)
\approx
\begin{bmatrix}
0.330&0.670
\end{bmatrix}
$$

所以：

$$
A\approx
\begin{bmatrix}
1&0\\
0.330&0.670
\end{bmatrix}
$$

每行之和为 $1$，表示每个 Query 如何分配读取权重。

### 3.5 对 Value 加权求和

输出：

$$
O=AV
$$

第一行：

$$
O_1
=
1\times
\begin{bmatrix}1&2\end{bmatrix}
+
0\times
\begin{bmatrix}1&3\end{bmatrix}
=
\begin{bmatrix}1&2\end{bmatrix}
$$

第二行：

$$
O_2
\approx
0.330
\begin{bmatrix}1&2\end{bmatrix}
+
0.670
\begin{bmatrix}1&3\end{bmatrix}
$$

$$
O_2
\approx
\begin{bmatrix}
1&2.670
\end{bmatrix}
$$

最终：

$$
O\approx
\begin{bmatrix}
1&2\\
1&2.670
\end{bmatrix}
$$

这说明第 2 个位置的新表示混合了第 1、2 个位置的 Value；第 1 个位置因果受限，只能读取自己。

## 4. 加回 batch 维

实际输入：

$$
X:[B,T,C]
$$

投影后：

$$
Q,K,V:[B,T,D]
$$

Key 转置最后两个维度：

$$
K^\mathsf{T}:[B,D,T]
$$

批量矩阵乘法：

$$
[B,T,D]\,[B,D,T]
\longrightarrow
[B,T,T]
$$

最后：

$$
[B,T,T]\,[B,T,D]
\longrightarrow
[B,T,D]
$$

第一个 $T$ 是 Query 位置，第二个 $T$ 是 Key 位置。混淆两者会让 mask 方向错误。

## 5. Padding Mask 与 Causal Mask

### 5.1 Causal Mask

控制“不能看未来”。它与序列中的时间顺序有关，通常屏蔽严格上三角。

### 5.2 Padding Mask

一个 batch 内序列长度不同，短序列用 PAD 补齐。Padding Mask 控制“不要读取补齐位置”。

例如真实长度为 $3$、补到 $5$：

$$
\begin{bmatrix}
x_1&x_2&x_3&\mathrm{PAD}&\mathrm{PAD}
\end{bmatrix}
$$

后两个位置不是真实内容。Key 侧应被屏蔽；计算损失时，PAD target 也应被 ignore。

### 5.3 两种 Mask 需要组合

在 decoder-only 模型中，一个位置既不能读取未来，也不能读取 PAD。组合时要小心广播维度：

$$
\text{causal mask}:[1,1,T,T]
$$

$$
\text{padding mask}:[B,1,1,T]
$$

组合后广播到：

$$
[B,H,T,T]
$$

## 6. 多头注意力为什么有用

单个注意力头只有一套投影和一张注意力分布。多头机制把模型维度拆成 $H$ 个子空间：

$$
C=H D_h
$$

每个头可学习不同投影：

$$
\operatorname{head}_h
=
\operatorname{Attention}
\left(
XW_Q^{(h)},
XW_K^{(h)},
XW_V^{(h)}
\right)
$$

所有头拼接：

$$
O_{\text{cat}}
=
\operatorname{Concat}
\left(
\operatorname{head}_1,\ldots,\operatorname{head}_H
\right)
$$

再经输出投影：

$$
O=O_{\text{cat}}W_O
$$

“一个头学语法、一个头学指代”是便于理解的可能现象，不是设计时强制指定的角色。头的功能由训练自行形成，也可能冗余。

## 7. 为什么需要位置编码

若没有位置编码，Self-Attention 只依赖 token 内容。对输入位置做同样的排列，输出也会按同样方式排列，模型不知道“谁在前、谁在后”。

### 7.1 可学习绝对位置

为每个位置准备向量：

$$
P\in\mathbb{R}^{T_{\max}\times C}
$$

输入变为：

$$
H_0=E_{\text{token}}+E_{\text{position}}
$$

简单直观，但超过训练最大位置时不能直接得到新位置向量。

### 7.2 正弦位置编码

原始 Transformer 使用：

$$
\operatorname{PE}(pos,2i)
=
\sin
\left(
\frac{pos}{10000^{2i/C}}
\right)
$$

$$
\operatorname{PE}(pos,2i+1)
=
\cos
\left(
\frac{pos}{10000^{2i/C}}
\right)
$$

不同维度对应不同频率，使位置具有可组合的周期表示。

### 7.3 RoPE 的核心直觉

RoPE 对 Query 和 Key 的二维分量施加与位置相关的旋转。点积因而自然包含相对位置差。第一遍学习不必推导全部复数形式，但要记住：它主要作用于 Query/Key，而不是简单给输入加一个位置向量。

## 8. LayerNorm 的原理

对单个 token 的 $C$ 个特征计算：

$$
\mu
=
\frac{1}{C}
\sum_{i=1}^{C}x_i
$$

$$
\sigma^2
=
\frac{1}{C}
\sum_{i=1}^{C}
(x_i-\mu)^2
$$

归一化并恢复可学习尺度：

$$
\operatorname{LN}(x_i)
=
\gamma_i
\frac{x_i-\mu}
{\sqrt{\sigma^2+\epsilon}}
+
\beta_i
$$

$\epsilon$ 防止方差接近零时除零。$\gamma,\beta$ 让网络保留重新缩放与平移的能力。

LayerNorm 与 BatchNorm 不同：它主要沿特征维处理单个 token，不依赖 batch 的统计量，更适合变长序列和自回归模型。

## 9. 残差连接为什么重要

残差结构：

$$
y=x+F(x)
$$

求导：

$$
\frac{\partial y}{\partial x}
=
I+\frac{\partial F}{\partial x}
$$

即使 $F$ 分支的梯度很小，仍存在恒等映射 $I$ 这条直接路径。这让深层网络更容易传递信息和梯度。

残差相加要求形状一致，所以 Transformer Block 通常保持：

$$
[B,T,C]\longrightarrow[B,T,C]
$$

## 10. MLP 在 Transformer 中做什么

Attention 在位置之间交换信息；MLP 对每个位置独立变换特征。典型形式：

$$
\operatorname{MLP}(x)
=
W_2\,\phi(W_1x+b_1)+b_2
$$

中间维度通常大于模型维度。以扩展倍数 $4$ 为例：

$$
[B,T,C]
\longrightarrow
[B,T,4C]
\longrightarrow
[B,T,C]
$$

Attention 回答“从哪些位置取信息”，MLP 负责“在当前 token 表示内部怎样加工这些信息”。

## 11. 一个 Pre-Norm Transformer Block

常见 decoder block：

$$
u
=
x+
\operatorname{Attention}
\left(
\operatorname{LayerNorm}(x)
\right)
$$

$$
y
=
u+
\operatorname{MLP}
\left(
\operatorname{LayerNorm}(u)
\right)
$$

输出仍为 $[B,T,C]$。堆叠多层后，每个位置可逐步形成更复杂的上下文表示。

## 12. Attention 复杂度

注意力分数矩阵大小为：

$$
[B,H,T,T]
$$

因此朴素 Attention 的时间和该矩阵内存通常随 $T^2$ 增长。上下文长度从 $T$ 翻倍到 $2T$ 时，分数元素数量约变为四倍：

$$
(2T)^2=4T^2
$$

这解释了为什么长上下文会显著增加训练和 prefill 成本。高效注意力实现可能避免显式保存完整矩阵或优化内存访问，但不会凭空消除所有计算。

## 13. 必做纸笔练习

1. 把第 3 节的 $W_V$ 改为单位矩阵，重新计算输出。
2. 去掉 causal mask，比较第 1 个位置是否读取了未来。
3. 构造三个 token、维度为 $2$ 的输入，写出 $QK^\mathsf{T}$ 形状。
4. 设 $B=2,H=4,T=8,D_h=16$，写出每个中间张量的形状。
5. 解释为什么 Softmax 必须沿 Key 维，而不是 Query 维。
6. 解释残差相加前为什么必须保持模型维度 $C$。

## 14. 代码实现前的检查表

- 每行注意力权重之和接近 $1$；
- causal mask 上三角权重为 $0$；
- PAD key 不会被真实 Query 读取；
- 所有张量无 NaN 和 Inf；
- 多头拆分前后元素总数不变；
- 输出形状与输入形状一致；
- 关闭 dropout 后，同一输入输出确定；
- 反向传播后投影矩阵均有有限梯度。
