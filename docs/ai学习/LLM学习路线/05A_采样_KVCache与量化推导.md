# 采样、KV Cache 与量化推导

## 讲义定位与使用方式

这是模块 5 的公式与手算补充。先理解主模块中的生成循环，再用这里的小数字验证采样、KV Cache 和量化；最后回到主模块完成性能实验。

## 1. 从 logits 到随机变量

模型最后一个位置输出：

$$
z\in\mathbb{R}^{V}
$$

Softmax 得到类别分布：

$$
p_i
=
\frac{e^{z_i}}
{\sum_{j=1}^{V}e^{z_j}}
$$

Greedy 选择：

$$
x_{t+1}=\arg\max_i p_i
$$

随机采样则是：

$$
x_{t+1}\sim\operatorname{Categorical}(p)
$$

Greedy 每步局部最优，不保证整条序列概率全局最优；采样可以探索多个合理续写。

## 2. Temperature 改变概率比

温度采样：

$$
p_i(T)
=
\frac{e^{z_i/T}}
{\sum_j e^{z_j/T}}
$$

两个 token 的概率比：

$$
\frac{p_i(T)}{p_j(T)}
=
e^{(z_i-z_j)/T}
$$

当 $T<1$，相同 logit 差被放大，分布更尖；当 $T>1$，差距被压缩。

例如 $z_i-z_j=2$：

$$
T=1:\quad e^2\approx7.39
$$

$$
T=0.5:\quad e^4\approx54.60
$$

$$
T=2:\quad e^1\approx2.72
$$

温度为零不能直接代入除法；程序应明确切换为 greedy。

## 3. Top-k 手算

假设五个 token 的概率为：

$$
p=
\begin{bmatrix}
0.40&0.25&0.20&0.10&0.05
\end{bmatrix}
$$

取 $k=3$，只保留前三项。重新归一化，保留概率和为：

$$
0.40+0.25+0.20=0.85
$$

新的分布：

$$
p'
\approx
\begin{bmatrix}
0.471&0.294&0.235&0&0
\end{bmatrix}
$$

不能过滤后直接使用旧概率而不归一化。

## 4. Top-p 手算

仍使用前面的降序概率。设 $p_{\text{nucleus}}=0.70$：

$$
0.40<0.70
$$

$$
0.40+0.25=0.65<0.70
$$

$$
0.40+0.25+0.20=0.85\ge0.70
$$

因此保留前三个 token。若下一步分布非常尖，例如：

$$
\begin{bmatrix}
0.80&0.10&0.05&0.03&0.02
\end{bmatrix}
$$

同样的 top-p 可能只保留第一个 token。这就是 top-p 候选集合大小会随上下文变化的原因。

## 5. Beam Search 分数

一条序列概率是各步条件概率乘积：

$$
P(x_{1:T})
=
\prod_{t=1}^{T}
P(x_t\mid x_{<t})
$$

实际使用对数分数：

$$
s(x_{1:T})
=
\sum_{t=1}^{T}
\log P(x_t\mid x_{<t})
$$

因为每个 log probability 通常非正，序列越长，累计分数往往越低，导致偏爱短序列。可使用长度归一化，例如：

$$
s_{\text{norm}}
=
\frac{s(x_{1:T})}{T^\alpha}
$$

$\alpha$ 需要针对任务验证。Beam width 增加会提高搜索成本，也不保证开放式文本更自然。

## 6. 不使用 KV Cache 的重复计算

生成第一个新 token 时，处理长度为 $T$ 的 prompt。生成下一个 token 时，若重新输入长度 $T+1$ 的全部序列，前 $T$ 个 token 的 Key 和 Value 会再次计算。

对每层：

$$
K_t=X_tW_K,\qquad V_t=X_tW_V
$$

历史 token 的 $K,V$ 在参数固定的推理过程中不会改变，因此可以缓存。

## 7. Prefill 与 Decode

### Prefill

一次处理整个 prompt：

$$
X:[B,T,C]
$$

建立每层 cache：

$$
K_{\text{cache}},V_{\text{cache}}
:[B,H_{\text{kv}},T,D_h]
$$

### Decode

新一步只输入一个 token：

$$
X_{\text{new}}:[B,1,C]
$$

计算新的 $Q,K,V$，把新 $K,V$ 追加到 cache。新 Query 与全部历史 Key 做点积：

$$
Q_{\text{new}}
K_{\text{cache}}^\mathsf{T}
$$

这样消除了历史 token 的重复投影与部分重复网络计算。

## 8. KV Cache 大小估算

若：

- 层数 $L$；
- batch $B$；
- KV head 数 $H_{\text{kv}}$；
- 序列长度 $T$；
- 每头维度 $D_h$；
- 每元素 $s$ 字节；
- Key 和 Value 两份。

缓存字节数近似：

$$
M_{\text{KV}}
=
2LBH_{\text{kv}}TD_hs
$$

例如：

$$
L=24,\quad B=1,\quad H_{\text{kv}}=8,
\quad T=4096,\quad D_h=128,\quad s=2
$$

则：

$$
M_{\text{KV}}
=
2\times24\times1\times8
\times4096\times128\times2
$$

约为 $384$ MiB。并发 batch 和上下文继续增加时，cache 很快成为服务瓶颈。

## 9. MHA、GQA 与 MQA 对 Cache 的影响

若 Query 有 $H$ 个头：

- MHA：$H_{\text{kv}}=H$；
- MQA：$H_{\text{kv}}=1$；
- GQA：$1<H_{\text{kv}}<H$。

由于 cache 大小与 $H_{\text{kv}}$ 近似成正比，共享 Key/Value 可以明显降低缓存，但模型质量和内核实现仍需实测。

## 10. 量化的仿射映射

给定浮点范围 $[x_{\min},x_{\max}]$，映射到整数范围 $[q_{\min},q_{\max}]$。Scale 可取：

$$
s
=
\frac{x_{\max}-x_{\min}}
{q_{\max}-q_{\min}}
$$

zero point：

$$
z
=
\operatorname{round}
\left(
q_{\min}-\frac{x_{\min}}{s}
\right)
$$

量化：

$$
q
=
\operatorname{clip}
\left(
\operatorname{round}\left(\frac{x}{s}\right)+z,
q_{\min},
q_{\max}
\right)
$$

反量化近似：

$$
\hat x=s(q-z)
$$

误差来自多个浮点值被映射到有限离散等级。

## 11. 对称量化例子

假设用有符号 8 bit 表示，整数范围约为 $[-127,127]$，浮点权重最大绝对值：

$$
\max|x|=2.54
$$

取：

$$
s=\frac{2.54}{127}=0.02
$$

浮点值 $x=0.73$ 被量化为：

$$
q=\operatorname{round}(0.73/0.02)=37
$$

反量化：

$$
\hat x=37\times0.02=0.74
$$

误差为 $0.01$。

## 12. Per-tensor、Per-channel 与 Group-wise

若整个矩阵共用一个 scale，极端值可能迫使多数小权重浪费量化等级。Per-channel 为每个输出通道使用独立 scale，group-wise 在小组内共享 scale，在精度、元数据和内核复杂度之间折中。

## 13. 为什么体积变小不保证加速

推理速度还取决于：

- 硬件是否支持相应整数/低位运算；
- 是否有高效反量化和矩阵乘内核；
- 模型是否受内存带宽还是计算限制；
- batch 和序列长度；
- 未量化层、cache 和数据搬运。

所以量化评估至少同时报告：模型文件、峰值内存、TTFT、TPOT、吞吐和质量指标。

## 14. 验证实验

1. 对固定 logits 分别应用 $T=0.5,1,2$，画概率柱状图。
2. 手算 top-k/top-p 后的归一化分布。
3. 对同一 prompt 比较无 cache 与 cache 的 greedy token 是否一致。
4. 按公式估算自己的模型在不同上下文长度下的 KV cache。
5. 对小矩阵做 INT8 量化，计算平均绝对误差和最大误差。
