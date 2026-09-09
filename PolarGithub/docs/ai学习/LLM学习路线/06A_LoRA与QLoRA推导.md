# LoRA、QLoRA 与监督微调推导

## 讲义定位与使用方式

这是模块 6 的推导补充。先在主模块理解数据、模板和评估，再用本讲义计算 LoRA/QLoRA 的参数与显存；手算结论要能解释回实际训练配置。

## 1. 微调在优化什么

预训练参数记为 $\theta_0$。全量微调直接寻找新参数：

$$
\theta^\star
=
\arg\min_\theta
L_{\text{task}}(\theta)
$$

LoRA 保持基础参数 $\theta_0$ 冻结，只训练一小组增量参数 $\phi$：

$$
\phi^\star
=
\arg\min_\phi
L_{\text{task}}(\theta_0,\phi)
$$

这降低了可训练参数、梯度和优化器状态，但不代表基础模型从内存中消失。

## 2. SFT 如何构造序列损失

对一轮对话：

$$
\text{system}\rightarrow
\text{user}\rightarrow
\text{assistant}
$$

Tokenizer 将模板变成一个 token 序列。设 assistant 回答从位置 $s$ 开始，可定义 mask：

$$
m_t=
\begin{cases}
0,&t<s\\
1,&t\ge s
\end{cases}
$$

损失：

$$
L
=
\frac{
\sum_t m_t
\left[
-\log P_\theta(x_{t+1}\mid x_{\le t})
\right]
}{
\sum_t m_t
}
$$

当 $m_t=0$ 时，system/user token 不参与目标。注意它们仍作为上下文输入，并非从序列删除。

## 3. LoRA 的低秩假设

全量权重：

$$
W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}
$$

LoRA 不直接学习同形状的任意增量，而是假设任务所需更新可由低秩矩阵近似：

$$
\Delta W=BA
$$

其中：

$$
A\in\mathbb{R}^{r\times d_{\text{in}}},
\qquad
B\in\mathbb{R}^{d_{\text{out}}\times r}
$$

因此：

$$
BA
\in
\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}
$$

前向传播：

$$
y
=
Wx+
\frac{\alpha}{r}BAx
$$

基础 $W$ 冻结，只更新 $A,B$。

## 4. 为什么叫“低秩”

矩阵 $BA$ 的秩最多为：

$$
\operatorname{rank}(BA)
\le
\min
\left(
\operatorname{rank}(A),
\operatorname{rank}(B)
\right)
\le r
$$

若 $r$ 远小于输入输出维度，$\Delta W$ 只能在有限子空间中变化。这是一种容量约束，也是一种参数压缩。

低秩假设并不保证所有任务都成立。若任务变化复杂、数据充足且算力允许，全量微调可能有更高上限。

## 5. 参数量手算

原权重参数量：

$$
N_{\text{full}}
=
d_{\text{out}}d_{\text{in}}
$$

LoRA 参数量：

$$
N_{\text{LoRA}}
=
r(d_{\text{in}}+d_{\text{out}})
$$

若：

$$
d_{\text{in}}=d_{\text{out}}=4096,
\qquad
r=16
$$

则原矩阵：

$$
N_{\text{full}}
=
4096^2
=
16{,}777{,}216
$$

LoRA：

$$
N_{\text{LoRA}}
=
16(4096+4096)
=
131{,}072
$$

比例约为：

$$
\frac{131{,}072}{16{,}777{,}216}
\approx0.78\%
$$

这是单个方阵的示例。整模型比例还取决于注入了哪些层。

## 6. $\alpha/r$ 缩放

LoRA 常使用：

$$
\Delta W
=
\frac{\alpha}{r}BA
$$

$r$ 改变时，直接使用 $BA$ 可能改变增量幅度。缩放使 rank 和更新尺度更容易分别调节。$\alpha$ 不是越大越好；过大可能使 adapter 主导基础模型输出，训练不稳定或遗忘原能力。

## 7. 初始化为什么常让一个矩阵为零

常见做法是随机初始化 $A$、将 $B$ 初始化为零，或反过来。这样训练开始时：

$$
\Delta W=BA=0
$$

模型初始行为与基础模型一致。若 $A,B$ 都为零，第一步两边可能同时得不到有效对称破缺，因此通常只把一边设零。

## 8. 注入哪些层

Attention 常见投影：

$$
W_Q,W_K,W_V,W_O
$$

MLP 也有上投影、门控投影和下投影。只给 $Q,V$ 加 LoRA 参数少；给所有线性层加 LoRA 容量更大、内存和训练成本也更高。

模块名称依模型代码而异。选择前应打印模型结构，确认目标模块实际存在，并统计可训练参数。

## 9. LoRA 合并

训练完成后可计算：

$$
W_{\text{merged}}
=
W+\frac{\alpha}{r}BA
$$

合并后推理不再单独计算 LoRA 分支。代价是每个 adapter 都需要一份合并模型，且失去运行时快速切换能力。

保存时应保留：

- 原始 adapter；
- adapter 配置；
- 精确基础模型名称与 revision；
- tokenizer 与 chat template；
- 训练数据和代码版本。

## 10. QLoRA 的计算路径

QLoRA 的典型思路：

1. 基础权重以 4 bit 形式存储；
2. 使用时按 group 的 scale 反量化到计算 dtype；
3. 与输入做矩阵运算；
4. LoRA 分支以更高精度参与计算；
5. 梯度只更新 LoRA 参数。

概念式：

$$
y
=
\operatorname{MatMul}
\left(
\operatorname{Dequant}(W_{4\text{bit}}),
x
\right)
+
\frac{\alpha}{r}BAx
$$

不能把它理解为“全部计算都是 4 bit”。实际计算 dtype 和内核由框架、硬件与配置决定。

## 11. NF4 的直觉

若权重分布近似以零为中心、接近正态，均匀量化不一定充分利用有限的 16 个 4-bit 代码。NF4 使用针对这种分布设计的非均匀量化点，让高概率区域拥有更细表示。

这是一种分布假设下的编码设计，不表示任意激活、任意权重都天然适合 NF4。

## 12. Double Quantization

分组量化需要为每组保存 scale。模型很大、group 很多时，scale 元数据也占空间。Double quantization 进一步量化这些量化常数，以降低平均每参数开销。

它节省的是元数据，并没有让主要权重从 4 bit 继续神奇地变为零成本。

## 13. 显存账本

LoRA/QLoRA 实验前至少估算：

$$
M_{\text{total}}
\approx
M_{\text{base}}
+M_{\text{adapter}}
+M_{\text{grad}}
+M_{\text{optimizer}}
+M_{\text{activation}}
+M_{\text{workspace}}
$$

QLoRA 主要降低 $M_{\text{base}}$。长序列导致的激活仍可能很大，所以序列长度、micro-batch 和 gradient checkpointing 依然关键。

## 14. 如何判断 rank 是否够用

固定数据、训练 token、学习率策略和目标层，只改变 $r$：

- 训练损失很高：容量可能不足，也可能数据/实现错误；
- 训练损失降低，验证不改善：增加 rank 未必有用，可能过拟合；
- 多个 rank 效果接近：优先较小者；
- 任务要求复杂风格与知识混合：先判断是否应加入 RAG，而不是无限增大 rank。

## 15. 必做验证

1. 训练前确认 $\Delta W$ 初始接近零；
2. 打印 trainable parameter ratio；
3. 检查一条样本的 chat template 和 mask；
4. 在 20 条数据上过拟合；
5. 对相同 prompts 比较 base 与 adapter；
6. 卸载 adapter 后输出应回到 base；
7. 合并前后在容差内输出一致；
8. 用保留测试集检查能力提升与灾难性遗忘。
