# 模块 2：NumPy 神经网络与 PyTorch

## 学习目标

先用 NumPy 看清“前向→损失→反向→更新”，再用 PyTorch 认识自动微分、Dataset、DataLoader、Module、Optimizer 和完整训练循环。

这一章不把神经网络当成几个 API 的组合。我们会从一个神经元开始，手算一次完整的前向传播和反向传播，然后再把手算过程翻译成 NumPy 和 PyTorch。读完后，你不仅应该能运行代码，还应该能回答：

- 权重和偏置各自在改变什么？
- 为什么连续堆叠线性层没有意义？
- Softmax 为什么要减去最大值？
- 交叉熵为什么是负对数？
- 反向传播究竟在“反向传播”什么？
- 为什么线性层的权重梯度要做转置和矩阵乘法？
- 为什么每一轮都要清空梯度？

---

## 学习导航

建议按下面的顺序学习：

1. 先用第 1 节的图像化说明建立直觉：一层层到底在传什么。
2. 跟着正文分别理解 Linear、ReLU、Softmax、交叉熵和反向传播；例子不是额外作业。
3. 学完第 9 节后，再看第 10 节用公式把多层 MLP 串起来。
4. 用第 20 节的 Checkpoint 内容检查训练流程；完成第 21 节“验收标准与知识复盘”后，最后再做第 22 节课后综合项目。

完整参考实现和文内示例的完整解答放在单独文件：[模块 2 标准答案](02_NumPy神经网络与PyTorch_标准答案.md)。

如果你已经理解各组件，想直接查看它们如何组成一份可运行程序，请打开：[NumPy 与 PyTorch MLP 完整模型代码](02_NumPy神经网络与PyTorch_完整模型代码.md)。

---

## 1. 多层 MLP 的整体直觉

### 1.1 多层 MLP 的“逐级加工流水线”类比

先不要急着看公式。把多层 MLP 想成一条处理信息的流水线：它接收一份原始信息，经过几次“提取特征、组合特征”，最后给出各个答案的分数。

以“识别手写数字”为例，一张 28×28 的图片可以先摊平成 784 个数字。单个数字本身没有“这是 8”的意思，只是每个位置有多亮。

```text
784 个像素数字
  ↓ 第一隐藏层：从原始像素中寻找简单线索
“这里像有一条竖线”“那里像有一个弯”
  ↓ 第二隐藏层：把简单线索组合成更复杂的形状
“像一个环”“像上下两个圆”
  ↓ 输出层：分别给 0～9 打分
“更像 8 的分数最高”
```

这只是帮助理解的比喻：某一个隐藏单元不保证真的只负责“竖线”或“圆”。重点是，后一层看到的不是原始像素，而是前一层已经整理过的一组“特征强弱”。

### 1.2 多层网络的层次结构

第一层把输入加工后产生一组数字；这组数字就是第二层的输入。第二层再次加工，产生新的数字；它们再交给输出层。

```text
原始输入
  → 第一层的输出（简单特征的强弱）
  → 第二层的输出（组合特征的强弱）
  → 输出层的分数（每个类别有多像）
```

所以你问的“上一层输出怎么变化应该怎么知道”，先从前向看很简单：

> 前一层的输出，就是后一层正在处理的输入。

不同的是，训练时还要反过来判断：这些中间特征让最终答案变好还是变坏？这个“反过来的反馈”就是反向传播；等学完各个零件后再用公式精确说明。

### 1.3 一次训练的完整信息流

把一次训练想成老师批改一份答卷：

1. **前向传播**：图片从第一层一路传到输出层，得到“像 0、1、…、9 的分数”。
2. **计算损失**：把预测和正确答案比较，判断这次错得多不多。
3. **反向传播**：从最后的错误开始，逐层追问“哪条连接让错误变大了，应该往哪个方向改一点”。
4. **更新参数**：SGD 或 Adam 根据这些建议，轻微调整每层的权重和偏置。

然后换下一张图片，重复很多次。网络不是一次就学会，而是靠大量“小幅纠正”慢慢形成有用的中间特征。

### 1.4 多层 MLP 的整体数据流

```text
输入数据
  ↓ 一层层加工（前向）
预测分数
  ↓ 和正确答案比较
损失：这次错了多少
  ↓ 把“该怎么改”的信息逐层传回去（反向）
每层参数的小改动
  ↓
下一次预测
```

现在不用背任何公式。接下来的第 2～9 节会分别解释每一个零件；到第 10 节再把它们按严格数学形式连成一个完整多层 MLP。

### 1.5 从直觉过渡到公式的学习顺序

公式版统一放在后面的第 10 节“多层 MLP 的完整前向与反向传播”。现在请直接进入第 2 节，从单个神经元开始。

---

## 2. 单个神经元的计算方式

### 2.1 单个神经元的加权打分机制

假设我们判断一封邮件是否是垃圾邮件，只使用两个特征：

- $x_1$：广告词出现次数；
- $x_2$：发件人是否在通讯录中，取值为 $0$ 或 $1$。

一个最简单的神经元先计算：

$$
z = w_1x_1 + w_2x_2 + b
$$

这里：

- $x_1,x_2$ 是输入；
- $w_1,w_2$ 是权重，表示模型对每个特征的重视程度；
- $b$ 是偏置，表示没有任何输入时模型原本的倾向；
- $z$ 是未经归一化的分数。

例如：

$$
x_1=3,\qquad x_2=1
$$

$$
w_1=0.8,\qquad w_2=-1.2,\qquad b=-0.5
$$

那么：

$$
z=0.8\times 3+(-1.2)\times 1-0.5=0.7
$$

正权重会把分数向上推，负权重会把分数向下拉。因为“广告词很多”支持垃圾邮件判断，所以 $w_1$ 可以为正；“发件人在通讯录中”反对垃圾邮件判断，所以 $w_2$ 可以为负。

这就是神经网络最基本的计算。所谓“训练”，就是根据大量样本自动寻找合适的 $w_1,w_2,b$。

### 2.2 偏置项提供可学习平移

若没有偏置：

$$
z=w_1x_1+w_2x_2
$$

当所有输入均为零时，输出只能为零。在线性分类的几何图像中，这意味着决策边界必须经过原点。加入偏置后，决策边界可以整体平移。

以二维情况为例，边界 $z=0$ 为：

$$
w_1x_1+w_2x_2+b=0
$$

改变权重主要改变直线的方向，改变偏置主要改变直线的位置。

### 2.3 参数与超参数

**参数**由训练数据学习，例如 $W$ 和 $b$。  
**超参数**由我们设定，例如学习率、隐藏层宽度、batch size 和训练轮数。

这一区别很重要：优化器更新参数，但不会自动决定你的网络应该有多少层。

---

## 3. 从单样本计算到批量矩阵计算

### 3.1 矩阵表示支持批量并行计算

假设一个 batch 有 $B$ 个样本，每个样本有 $D_{\text{in}}$ 个输入特征，希望产生 $D_{\text{out}}$ 个输出分数。把所有样本排成矩阵：

$$
X\in\mathbb{R}^{B\times D_{\text{in}}}
$$

权重和偏置为：

$$
W\in\mathbb{R}^{D_{\text{in}}\times D_{\text{out}}},
\qquad
b\in\mathbb{R}^{D_{\text{out}}}
$$

线性层为：

$$
Z=XW+b
$$

输出形状是：

$$
Z\in\mathbb{R}^{B\times D_{\text{out}}}
$$

偏置 $b$ 会沿 batch 维广播，相当于给每个样本都加同一个偏置向量。

这里不是把不同样本混在一起，而是将形状为 $(D_{\text{out}},)$ 的偏置向量 $b$ 从右侧对齐为 $(1,D_{\text{out}})$，再沿 Batch 维重复使用。广播的完整判断规则、行向量与列向量的区别，以及 `keepdims=True` 的示例见 [模块 1 的广播详解](01_Python_Git_数学基础.md)。

### 3.2 线性层的手算示例

设有两个样本，每个样本两个特征，输出三个类别的分数：

$$
X=
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix},
\qquad
W=
\begin{bmatrix}
1 & 0 & -1\\
0 & 2 & 1
\end{bmatrix},
\qquad
b=
\begin{bmatrix}
0.5 & -0.5 & 1
\end{bmatrix}
$$

先计算第一条样本：

$$
\begin{bmatrix}1&2\end{bmatrix}W
=
\begin{bmatrix}
1\times 1+2\times 0 &
1\times 0+2\times 2 &
1\times(-1)+2\times1
\end{bmatrix}
=
\begin{bmatrix}1&4&1\end{bmatrix}
$$

再加偏置：

$$
\begin{bmatrix}1&4&1\end{bmatrix}
+
\begin{bmatrix}0.5&-0.5&1\end{bmatrix}
=
\begin{bmatrix}1.5&3.5&2\end{bmatrix}
$$

三个数字不是概率，而是三个类别的 **logits**。Logit 可以是任意实数，也不需要相加等于 $1$。

### 3.3 矩阵乘法不是逐元素相乘

矩阵乘法 $XW$ 会对输入特征做加权求和；逐元素乘法 $X\odot W$ 只把同位置元素相乘。二者表达的运算完全不同。

判断矩阵乘法形状时使用“内维相等、外维保留”：

$$
(B\times D_{\text{in}})
(D_{\text{in}}\times D_{\text{out}})
\longrightarrow
(B\times D_{\text{out}})
$$

如果代码报矩阵形状错误，先把每个维度的语义写出来，不要靠反复交换转置碰运气。

---

## 4. 激活函数与非线性表达能力

### 4.1 两个线性层仍然只是一个线性层

如果没有激活函数：

$$
H=XW_1+b_1
$$

$$
Z=HW_2+b_2
$$

代入 $H$：

$$
Z=(XW_1+b_1)W_2+b_2
$$

整理得到：

$$
Z=X(W_1W_2)+(b_1W_2+b_2)
$$

令：

$$
W'=W_1W_2,\qquad b'=b_1W_2+b_2
$$

就有：

$$
Z=XW'+b'
$$

所以无论堆叠多少个纯线性层，都可以合并成一个线性层，仍然只能表示线性关系。层数变多了，表达能力却没有本质增加。

### 4.2 ReLU 引入非线性表达能力

ReLU 定义为：

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

它把负数截断为零，正数保持不变：

$$
\operatorname{ReLU}(-2)=0,\qquad
\operatorname{ReLU}(3)=3
$$

经过 ReLU 后，不同输入区域会使用不同的线性关系，多个区域拼接起来就能形成弯曲、分段线性的决策边界。

其导数在 $z\ne 0$ 时为：

$$
\frac{\partial\operatorname{ReLU}(z)}{\partial z}
=
\begin{cases}
0,&z<0\\
1,&z>0
\end{cases}
$$

在 $z=0$ 处数学上不可导，深度学习框架通常约定使用某个次梯度，常见实现取 $0$。单个点的选择通常不会影响实际训练。

### 4.3 ReLU 的局限性

若某个神经元长期处在负半轴，它的梯度一直为零，可能成为“死亡 ReLU”。常见缓解方式包括更合理的初始化、合适的学习率，或使用 Leaky ReLU、GELU 等激活函数。

---

## 5. 从 Logits 到概率：Softmax

### 5.1 Logits 不满足概率分布约束

概率必须满足：

$$
p_i\ge 0,\qquad \sum_{i=1}^{C}p_i=1
$$

而 logits 可能是负数，也不要求总和为 $1$。Softmax 将它们转换为概率：

$$
p_i=
\frac{\exp(z_i)}
{\sum_{j=1}^{C}\exp(z_j)}
$$

指数函数保证结果为正，除以总和保证概率之和为 $1$。

### 5.2 三分类 Softmax 手算示例

给定：

$$
z=
\begin{bmatrix}
2&1&0
\end{bmatrix}
$$

则：

$$
\exp(z)=
\begin{bmatrix}
e^2&e^1&e^0
\end{bmatrix}
\approx
\begin{bmatrix}
7.389&2.718&1
\end{bmatrix}
$$

总和约为 $11.107$，因此：

$$
p\approx
\begin{bmatrix}
0.665&0.245&0.090
\end{bmatrix}
$$

最大的 logit 对应最大的概率，但概率大小还取决于各 logit 之间的差距。

### 5.3 减去最大值提高数值稳定性

指数增长很快，$\exp(1000)$ 会溢出。但 Softmax 对所有 logits 同时减去常数 $c$ 后结果不变：

$$
\frac{\exp(z_i-c)}
{\sum_j\exp(z_j-c)}
=
\frac{\exp(z_i)/\exp(c)}
{\sum_j\exp(z_j)/\exp(c)}
=
\frac{\exp(z_i)}
{\sum_j\exp(z_j)}
$$

因此选择：

$$
c=\max_j z_j
$$

可以让最大的指数变为 $\exp(0)=1$，显著提高数值稳定性。

---

这段代码定义了一个 **Softmax 函数**：

```python
def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)
```

Softmax 的作用是：把一组任意实数转换成一组 **0 到 1 之间、总和为 1** 的数，通常可以理解为概率分布。

例如：

```python
x = np.array([1.0, 2.0, 3.0])
```

经过 Softmax 后，大致得到：

```
[0.0900, 0.2447, 0.6652]
```

它们都大于 0，并且：

```
0.0900 + 0.2447 + 0.6652 ≈ 1
```

---

### 5.4 Softmax 的稳定 NumPy 实现

**定义 Softmax 函数**

```python
def softmax(x, axis=-1):
```

这里定义了一个名叫 `softmax` 的函数。

它有两个参数：

- `x`：需要处理的数组
- `axis=-1`：默认沿最后一个维度计算 Softmax

例如 `x` 是二维数组：

```python
x.shape == (2, 3)
```

那么 `axis=-1` 就是沿每一行的 3 个元素计算。

---

**找到每行最大值并减去**

```python
shifted = x - np.max(x, axis=axis, keepdims=True)
```

先看里面：

```python
np.max(x, axis=axis, keepdims=True)
```

它会沿指定维度找到最大值。

例如：

```python
x = np.array([
    [1, 2, 3],
    [4, 8, 6]
])
```

沿最后一个维度找最大值：

```python
np.max(x, axis=-1, keepdims=True)
```

得到：

```
[[3],
 [8]]
```

然后用每一行减去自己这一行的最大值：

```
[[1, 2, 3],       [[3],       [[-2, -1,  0],
 [4, 8, 6]]   -    [8]]   =    [-4,  0, -2]]
```

所以：

```python
shifted
```

就是：

```
[[-2, -1,  0],
 [-4,  0, -2]]
```

**减去最大值的数学依据与数值作用**

因为后面要计算指数：

```python
np.exp(x)
```

如果 `x` 很大，比如：

```python
x = 1000
```

那么：

```python
np.exp(1000)
```

会大到超出计算机浮点数的范围，造成溢出。

减去最大值之后，最大的元素一定变为 `0`，其他元素小于或等于 `0`：

```
[-2, -1, 0]
```

指数就变成：

```
[e^-2, e^-1, e^0]
```

其中最大的只是：

```
e^0 = 1
```

这样计算更加稳定。

而减去相同的数不会改变最终 Softmax 结果，因为：

$$
\frac{e^{x_i-c}}{\sum_j e^{x_j-c}}
=
\frac{e^{x_i}e^{-c}}{e^{-c}\sum_j e^{x_j}}
=
\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

分子和分母中的 $e^{-c}$ 会相互约掉。

---

**对平移后的 Logits 求指数**

```python
exp = np.exp(shifted)
```

`np.exp()` 计算的是自然指数：

```
e^x
```

例如：

```python
shifted = np.array([-2, -1, 0])
```

那么：

```python
np.exp(shifted)
```

大约得到：

```
[0.1353, 0.3679, 1.0000]
```

所以变量 `exp` 保存的是每个元素的指数值。

这里的变量名 `exp` 不会影响 `np.exp`：

```python
exp = np.exp(shifted)
```

右边的 `np.exp` 是 NumPy 函数，左边的 `exp` 只是自己起的变量名。

---

**用指数和完成归一化**

```python
return exp / exp.sum(axis=axis, keepdims=True)
```

先计算：

```python
exp.sum(axis=axis, keepdims=True)
```

假设：

```
exp = [0.1353, 0.3679, 1.0000]
```

那么总和是：

```
0.1353 + 0.3679 + 1.0000 = 1.5032
```

再让每个元素除以总和：

```
0.1353 / 1.5032 ≈ 0.0900
0.3679 / 1.5032 ≈ 0.2447
1.0000 / 1.5032 ≈ 0.6652
```

最终得到：

```
[0.0900, 0.2447, 0.6652]
```

它们的总和为 1。

---

**`keepdims=True` 保留广播所需维度**

它表示：求最大值或求和之后，**保留原来的维度数量**。

例如：

```python
x = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
```

它的形状是：

```
(2, 3)
```

不使用 `keepdims=True`：

```python
np.max(x, axis=-1)
```

结果是：

```
[3, 6]
```

形状为：

```
(2,)
```

使用：

```python
np.max(x, axis=-1, keepdims=True)
```

结果是：

```
[[3],
 [6]]
```

形状为：

```
(2, 1)
```

保留成 `(2, 1)` 后，NumPy 能通过广播机制，让每一行减去对应的最大值：

```
[[1, 2, 3] - [3],
 [4, 5, 6] - [6]]
```

同样，求和时保留维度，也方便每一行除以各自的总和。

---

**Softmax 完整实现示例**

```python
import numpy as np

def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)

x = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 2.0, 2.0]
])

result = softmax(x)

print(result)
print(result.sum(axis=-1))
```

输出大致是：

```
[[0.09003057 0.24472847 0.66524096]
 [0.33333333 0.33333333 0.33333333]]

[1. 1.]
```

第一行中，数字越大，Softmax 后的概率越大；第二行三个数字相同，所以得到三个相同的概率：

```
1/3, 1/3, 1/3
```

整段代码可以概括为：

> 每组数据先减去这一组的最大值，再取指数，最后除以所有指数之和，将数据转换为总和为 1 的概率分布。

## 6. 交叉熵损失的原理

### 6.1 从最大似然理解

若真实类别是 $y$，模型给它的概率为 $p_y$。我们希望 $p_y$ 越大越好，即最大化所有训练样本正确类别概率的乘积：

$$
\max_\theta \prod_{n=1}^{N}p_\theta(y_n\mid x_n)
$$

许多小概率相乘容易数值下溢，而且乘积难求导。取对数后，乘积变成求和：

$$
\max_\theta
\sum_{n=1}^{N}
\log p_\theta(y_n\mid x_n)
$$

优化器通常执行最小化，于是取负号：

$$
L=
-\frac{1}{N}
\sum_{n=1}^{N}
\log p_\theta(y_n\mid x_n)
$$

这就是 one-hot 分类目标下的平均交叉熵。

### 6.2 损失值反映模型的置信程度

如果模型给正确类别的概率分别为：

$$
p_y=0.9,\quad 0.5,\quad 0.1
$$

对应损失约为：

$$
-\log 0.9\approx0.105
$$

$$
-\log 0.5\approx0.693
$$

$$
-\log 0.1\approx2.303
$$

模型对错误答案越自信，惩罚增长越快。这就是交叉熵比单纯“对或错”包含更多训练信号的原因。

### 6.3 PyTorch 交叉熵直接接收 Logits

PyTorch 的 CrossEntropyLoss 接收原始 logits，内部使用数值稳定的 log-softmax 与负对数似然组合。如果先手工 Softmax：

- 会重复计算；
- 更容易出现概率下溢；
- 输入语义不再符合函数要求；
- 梯度可能变得更差。

---

这页想表达的是：

> **使用 `torch.nn.CrossEntropyLoss` 时，模型最后一层不要自己写 `Softmax`，直接把原始输出 `logits` 交给它。**

先看标准写法：

```python
logits = model(x)
loss = criterion(logits, target)
```

而不要写成：

```python
probabilities = torch.softmax(model(x), dim=1)
loss = criterion(probabilities, target)
```

**CrossEntropyLoss：什么是 logits**

假设模型要把一张图片分成三类，最后输出：

```python
logits = [2.0, 1.0, 0.1]
```

这些数叫 **logits**。

它们不是概率：

- 可以大于 1
- 可以小于 0
- 总和不需要等于 1

但它们表示模型对各类别的原始评分。这里第一类得分最高，所以模型更倾向第一类。

如果对它们做 Softmax：

```python
softmax([2.0, 1.0, 0.1])
```

大约得到：

```
[0.659, 0.242, 0.099]
```

这才是概率。

**CrossEntropyLoss：内部已经做了什么**

PyTorch 的：

```python
torch.nn.CrossEntropyLoss()
```

内部可以大致理解为：

```python
log_softmax + NLLLoss
```

也就是它会自己：

1. 把 logits 转成对数概率；
2. 取正确类别对应的值；
3. 计算损失。

所以：

```python
loss = CrossEntropyLoss(logits, target)
```

已经包含了 Softmax 相关计算。

不过它不是先显式计算普通 Softmax，再取对数，而是直接使用数值更稳定的 `log_softmax`。

**提前 Softmax 会造成重复归一化**

假设你写：

```python
probabilities = torch.softmax(logits, dim=1)
loss = criterion(probabilities, target)
```

此时 `CrossEntropyLoss` 不知道你传入的是概率。

它仍然会把这些概率当成新的 logits，再执行一次 `log_softmax`。

例如原始 logits：

```
[2.0, 1.0, 0.1]
```

正确的 Softmax：

```
[0.659, 0.242, 0.099]
```

你把这个概率传入 `CrossEntropyLoss` 后，它又会把：

```
[0.659, 0.242, 0.099]
```

当成原始评分，再做一次 Softmax。新的结果大约是：

```
[0.478, 0.315, 0.207]
```

可以看到，原本第一类的概率是 `0.659`，结果变成了 `0.478`。

信息被扭曲了。

因此图片里说的“会重复计算”，更准确地说是：

> 你先做了一次 Softmax，而 `CrossEntropyLoss` 又把结果当 logits 做了一次类似的归一化。

**数值稳定计算可以避免溢出与下溢**

普通 Softmax 的公式是：

$$
p_i=\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

如果 logits 很大，例如：

```
[1000, 999, 998]
```

直接计算：

```
e^1000
```

会特别大，可能超出浮点数范围。

如果概率非常小，例如：

```
0.00000000000000000001
```

再计算：

```python
torch.log(probability)
```

也容易出现精度问题，极端情况下可能变成：

```
log(0) = -∞
```

`CrossEntropyLoss` 内部使用稳定的 `log_softmax`，会通过减去最大值等方法避免这些问题。

原始 logits：

```
[1000, 999, 998]
```

先减去最大值：

```
[0, -1, -2]
```

这样再计算指数就安全很多。

**`CrossEntropyLoss` 的输入语义要求**

`CrossEntropyLoss` 要求输入表示：

```
每个类别的原始分数 logits
```

但 Softmax 后的数据表示：

```
每个类别的概率
```

它们含义不同。

所以虽然它们都是一组数字，形状也可能完全相同，但函数期待的是 logits，不是概率。

就像某个函数要求输入摄氏度，你却提前转换成华氏度；数字仍然能传进去，但函数会按照摄氏度解释，结果自然不对。

**提前 Softmax 对梯度质量的影响**

训练时，PyTorch 需要根据损失计算梯度，告诉模型参数应该怎么调整。

正确过程：

```
logits
  ↓
CrossEntropyLoss
  ↓
稳定地得到损失和梯度
```

提前 Softmax 后：

```
logits
  ↓
Softmax
  ↓
概率
  ↓
CrossEntropyLoss 又当成 logits 处理
```

多出来的 Softmax 会压缩数值差异。

例如：

```
logits = [10, 0, -10]
```

差异很明显。

Softmax 后可能接近：

```
[0.99995, 0.00005, 0.00000]
```

很多概率已经非常接近 0 或 1，Softmax 在这些区域的梯度可能很小，使参数更新变慢。

**CrossEntropyLoss：正确代码**

多分类任务通常这样写：

```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

logits = model(x)          # 不要 Softmax
loss = criterion(logits, target)
```

其中：

```python
logits.shape
```

通常是：

```
(batch_size, 类别数量)
```

而 `target` 是每个样本正确类别的编号：

```python
target = torch.tensor([2, 0, 1])
```

不是 one-hot 概率。

**CrossEntropyLoss：什么时候需要 Softmax**

**训练计算损失时：**

```python
loss = criterion(logits, target)
```

不要先 Softmax。

**训练结束后，想查看概率时：**

```python
probabilities = torch.softmax(logits, dim=1)
```

可以使用 Softmax。

例如：

```python
logits = model(x)

loss = criterion(logits, target)

probabilities = torch.softmax(logits, dim=1)
predicted_class = probabilities.argmax(dim=1)
```

其实只想知道预测类别时，甚至不需要 Softmax：

```python
predicted_class = logits.argmax(dim=1)
```

因为 Softmax 不会改变数字的大小顺序：

```
logits 最大的位置
=
Softmax 概率最大的位置
```

一句话记忆：

> **训练时把 logits 直接交给 `CrossEntropyLoss`；展示概率时再使用 Softmax。**

## 7. 训练目标与梯度下降

### 7.1 损失函数的地形图类比

模型的所有参数组成一个高维坐标 $\theta$。每一组参数都对应一个损失值 $L(\theta)$。训练的目标是在这张高维地形图上找到较低的位置。

梯度：

$$
\nabla_\theta L
$$

指向损失上升最快的方向，所以梯度下降沿反方向更新：

$$
\theta_{t+1}
=
\theta_t-\eta\nabla_\theta L
$$

其中 $\eta$ 是学习率。

### 7.2 学习率控制参数更新幅度

- 学习率太小：每步移动很少，训练很慢；
- 学习率适中：损失稳定下降；
- 学习率太大：越过低谷，损失振荡甚至变成 NaN。

学习率不是越小越安全。若训练预算固定，过小可能根本来不及学到有效参数。

### 7.3 Batch、Iteration 与 Epoch

- **样本**：一条训练数据；
- **batch**：一次前向和反向使用的一组样本；
- **iteration/step**：完成一次参数更新；
- **epoch**：大致完整遍历一次训练集。

若训练集有 $1000$ 条数据，batch size 为 $100$，那么一个 epoch 大约有 $10$ 个 step。

mini-batch 梯度是完整数据梯度的带噪声估计。噪声并非总是坏事，它能降低每步计算成本，有时还帮助优化器离开不理想区域。

---

## 8. 链式法则与反向传播

### 8.1 标量计算图中的链式法则

设：

$$
a=wx+b
$$

$$
\hat y=a^2
$$

$$
L=(\hat y-y)^2
$$

计算图是：

$$
(w,x,b)\longrightarrow a\longrightarrow \hat y\longrightarrow L
$$

反向传播从最后的损失开始，逐节点计算“损失对该节点的敏感度”。

由链式法则：

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat y}
\frac{\partial \hat y}{\partial a}
\frac{\partial a}{\partial w}
$$

分别计算：

$$
\frac{\partial L}{\partial \hat y}
=2(\hat y-y)
$$

$$
\frac{\partial \hat y}{\partial a}=2a
$$

$$
\frac{\partial a}{\partial w}=x
$$

所以：

$$
\frac{\partial L}{\partial w}
=2(\hat y-y)\cdot2a\cdot x
$$

反向传播传递的不是“答案”，而是上游梯度，即损失对中间量的导数。

### 8.2 多路径梯度需要累加

一个变量可能通过多条路径影响损失。若：

$$
L=f(x)+g(x)
$$

那么：

$$
\frac{\partial L}{\partial x}
=
\frac{\partial f}{\partial x}
+
\frac{\partial g}{\partial x}
$$

因此自动微分框架会把来自不同路径的梯度累加。PyTorch 多次调用 backward 时也会继续累加到参数的 grad 字段，所以每个训练 step 前要清梯度。

---

## 9. 线性层反向传播的矩阵推导

这是从“会调用 backward”走向“真正理解 Transformer 代码”的关键一节。

### 9.1 线性层反向传播的符号与形状

前向传播：

$$
Z=XW+b
$$

形状为：

$$
X:[B,D_{\text{in}}],
\quad
W:[D_{\text{in}},D_{\text{out}}],
\quad
Z:[B,D_{\text{out}}]
$$

假设后续网络已经传回：

$$
G=\frac{\partial L}{\partial Z}
$$

其形状与 $Z$ 相同：

$$
G:[B,D_{\text{out}}]
$$

我们要得到对 $X,W,b$ 的梯度。

### 9.2 对权重的梯度

单个输出元素为：

$$
Z_{n,j}
=
\sum_{i=1}^{D_{\text{in}}}
X_{n,i}W_{i,j}+b_j
$$

固定某个权重 $W_{i,j}$，它会影响 batch 中每个样本的 $Z_{n,j}$。根据链式法则：

$$
\frac{\partial L}{\partial W_{i,j}}
=
\sum_{n=1}^{B}
\frac{\partial L}{\partial Z_{n,j}}
\frac{\partial Z_{n,j}}{\partial W_{i,j}}
$$

又因为：

$$
\frac{\partial Z_{n,j}}{\partial W_{i,j}}=X_{n,i}
$$

所以：

$$
\frac{\partial L}{\partial W_{i,j}}
=
\sum_{n=1}^{B}
X_{n,i}G_{n,j}
$$

写成矩阵：

$$
\boxed{
\frac{\partial L}{\partial W}
=X^\mathsf{T}G
}
$$

形状检查：

$$
[D_{\text{in}},B]
[B,D_{\text{out}}]
\longrightarrow
[D_{\text{in}},D_{\text{out}}]
$$

结果恰好与 $W$ 同形状。

### 9.3 对偏置的梯度

同一个偏置 $b_j$ 被 batch 内所有样本共享，因此要沿 batch 维求和：

$$
\boxed{
\frac{\partial L}{\partial b_j}
=
\sum_{n=1}^{B}G_{n,j}
}
$$

矩阵写法是：

$$
\boxed{
\frac{\partial L}{\partial b}
=
\operatorname{sum}(G,\text{axis}=0)
}
$$

**权重与偏置共享同一个上游梯度**

对，**计算权重梯度和偏置梯度时，用到的是同一个上游梯度 $G$**，但最后得到的两个梯度并不一样。

这里：

$$
G_{n,j}=\frac{\partial L}{\partial Z_{n,j}}
$$

表示损失 $L$ 对线性层输出 $Z$ 的梯度。

线性层是：

$$
Z=XW+b
$$

所以 $W$ 和 $b$ 都通过同一个 $Z$ 影响损失 $L$。反向传播时，梯度从 $L$ 传到 $Z$，先得到同一个：

$$
G=\frac{\partial L}{\partial Z}
$$

然后在 $Z=XW+b$ 这个位置分成三路：

$$
G
\longrightarrow
\frac{\partial L}{\partial W},
\quad
\frac{\partial L}{\partial b},
\quad
\frac{\partial L}{\partial X}
$$

**用链式法则分别沿两条路径计算**

对于单个输出元素：

$$
Z_{n,j}=\sum_i X_{n,i}W_{i,j}+b_j
$$

**权重路径：$W_{i,j}$**

根据链式法则：

$$
\frac{\partial L}{\partial W_{i,j}}
=
\sum_n
\frac{\partial L}{\partial Z_{n,j}}
\frac{\partial Z_{n,j}}{\partial W_{i,j}}
$$

其中：

$$
\frac{\partial L}{\partial Z_{n,j}}=G_{n,j}
$$

而：

$$
\frac{\partial Z_{n,j}}{\partial W_{i,j}}=X_{n,i}
$$

所以：

$$
\frac{\partial L}{\partial W_{i,j}}
=
\sum_n X_{n,i}G_{n,j}
$$

写成矩阵：

$$
\frac{\partial L}{\partial W}=X^T G
$$

**偏置路径：$b_j$**

同样：

$$
\frac{\partial L}{\partial b_j}
=
\sum_n
\frac{\partial L}{\partial Z_{n,j}}
\frac{\partial Z_{n,j}}{\partial b_j}
$$

仍然有：

$$
\frac{\partial L}{\partial Z_{n,j}}=G_{n,j}
$$

但是：

$$
\frac{\partial Z_{n,j}}{\partial b_j}=1
$$

因此：

$$
\frac{\partial L}{\partial b_j}
=
\sum_n G_{n,j}
$$

矩阵写法：

```python
grad_b = G.sum(axis=0)
```

**同一上游梯度沿不同局部路径产生不同结果**

因为链式法则里除了上游梯度 $G$，还要乘上当前运算的局部梯度。

权重路径：

$$
\frac{\partial L}{\partial W}
=
\underbrace{\frac{\partial L}{\partial Z}}_{G}
\cdot
\underbrace{\frac{\partial Z}{\partial W}}_{\text{与 }X\text{ 有关}}
$$

偏置路径：

$$
\frac{\partial L}{\partial b}
=
\underbrace{\frac{\partial L}{\partial Z}}_{G}
\cdot
\underbrace{\frac{\partial Z}{\partial b}}_{1}
$$

所以：

- 权重梯度要乘输入 $X$；
- 偏置梯度只需要把 $G$ 在 batch 维度相加。

**用标量例子检查直觉**

假设：

$$
z=xw+b
$$

损失是 $L$，并且上游已经给出：

$$
\frac{\partial L}{\partial z}=g
$$

那么：

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial z}
\frac{\partial z}{\partial w}
=
g\cdot x
$$

而：

$$
\frac{\partial L}{\partial b}
=
\frac{\partial L}{\partial z}
\frac{\partial z}{\partial b}
=
g\cdot 1
=
g
$$

你可以看到，两者都从同一个 $g$ 出发，但：

$$
\frac{\partial z}{\partial w}=x
$$

而：

$$
\frac{\partial z}{\partial b}=1
$$

所以结果不同。

**对应到代码**

```python
def backward(self, grad_out):
    self.grad_W = self.x.T @ grad_out
    self.grad_b = grad_out.sum(axis=0)
    return grad_out @ self.W.T
```

这里的：

```python
grad_out
```

就是：

$$
G=\frac{\partial L}{\partial Z}
$$

同一个 `grad_out` 被用于三项计算：

```python
self.x.T @ grad_out       # 得到 ∂L/∂W
grad_out.sum(axis=0)      # 得到 ∂L/∂b
grad_out @ self.W.T       # 得到 ∂L/∂X
```

它们共享同一个上游梯度，但各自乘上不同的局部导数。

一句话概括：

> $W$、$b$ 和 $X$ 都通过同一个输出 $Z$ 影响损失，所以反向传播都从同一个 $G=\partial L/\partial Z$ 开始；但由于 $Z$ 对它们的局部导数不同，最终梯度也不同。

### 9.4 对输入的梯度

输入还可能来自前一层，因此必须继续把梯度传回去：

$$
\boxed{
\frac{\partial L}{\partial X}
=GW^\mathsf{T}
}
$$

形状检查：

$$
[B,D_{\text{out}}]
[D_{\text{out}},D_{\text{in}}]
\longrightarrow
[B,D_{\text{in}}]
$$

### 9.5 形状检查用于定位矩阵求导错误

一个变量的梯度必须与变量本身形状相同：

$$
\operatorname{shape}
\left(
\frac{\partial L}{\partial W}
\right)
=
\operatorname{shape}(W)
$$

如果得到的 grad*W 是 $[D*{\text{out}},D\_{\text{in}}]$，基本可以断定转置位置错了。

---

## 10. 多层 MLP 的完整前向与反向传播

到这里，你已经分别认识了线性层、ReLU、Softmax、交叉熵和单层反向传播。现在再回头看多层 MLP：它没有新的魔法，只是把你已经学会的零件按顺序连接起来。

上一节只推导了一层：

$$
Z=XW+b
$$

但多层 MLP 只是把这个结构重复多次。关键点是：

> 每一层不仅要计算自己的参数梯度，还要计算“损失对本层输入的梯度”，并把它传给前一层。

这就是：

$$
\frac{\partial L}{\partial X}=GW^\mathsf{T}
$$

为什么重要。

### 10.1 三层 MLP 的前向传播

设：

$$
A_0=X
$$

隐藏层 1：

$$
Z_1=A_0W_1+b_1
$$

$$
A_1=ReLU(Z_1)
$$

隐藏层 2：

$$
Z_2=A_1W_2+b_2
$$

$$
A_2=ReLU(Z_2)
$$

输出层：

$$
Z_3=A_2W_3+b_3
$$

$$
logits=Z_3
$$

代码上就是：

```python
z1 = x @ W1 + b1
a1 = relu(z1)

z2 = a1 @ W2 + b2
a2 = relu(z2)

logits = a2 @ W3 + b3
loss = cross_entropy(logits, y)
```

假设一个 `batch` 有 $B$ 个样本，输入维度为 $D_{\text{in}}$，两层隐藏层宽度是 $H_1,H_2$，类别数是 $C$，那么形状应一路这样变化：

| 变量         |                形状 | 用人话说                 |
| ------------ | ------------------: | ------------------------ |
| $A_0=X$      | $[B,D_{\text{in}}]$ | 一批原始输入             |
| $Z_1,A_1$    |           $[B,H_1]$ | 第一层提取出的特征       |
| $Z_2,A_2$    |           $[B,H_2]$ | 第二层组合后的特征       |
| $Z_3=logits$ |             $[B,C]$ | 每个样本对每个类别的分数 |

这就是“一层层向前传播”。每一层的输出会成为下一层的输入。

### 10.2 三层 MLP 的反向传播

假设交叉熵已经给出：

$$
G_3=\frac{\partial L}{\partial Z_3}
$$

输出层反向：

$$
\frac{\partial L}{\partial W_3}=A_2^\mathsf{T}G_3
$$

$$
\frac{\partial L}{\partial b_3}=\operatorname{sum}(G_3,\text{axis}=0)
$$

$$
\frac{\partial L}{\partial A_2}=G_3W_3^\mathsf{T}
$$

注意最后一行：它就是“上一层输出 $A_2$ 对最终 loss 的影响”。

接着穿过 ReLU：

$$
G_2
=
\frac{\partial L}{\partial Z_2}
=
\frac{\partial L}{\partial A_2}\odot \mathbf{1}[Z_2>0]
$$

第二层线性层反向：

$$
\frac{\partial L}{\partial W_2}=A_1^\mathsf{T}G_2
$$

$$
\frac{\partial L}{\partial b_2}=\operatorname{sum}(G_2,\text{axis}=0)
$$

$$
\frac{\partial L}{\partial A_1}=G_2W_2^\mathsf{T}
$$

再穿过第一层 ReLU：

$$
G_1
=
\frac{\partial L}{\partial Z_1}
=
\frac{\partial L}{\partial A_1}\odot \mathbf{1}[Z_1>0]
$$

第一层线性层反向：

$$
\frac{\partial L}{\partial W_1}=A_0^\mathsf{T}G_1
$$

$$
\frac{\partial L}{\partial b_1}=\operatorname{sum}(G_1,\text{axis}=0)
$$

如果还需要继续往输入传：

$$
\frac{\partial L}{\partial A_0}=G_1W_1^\mathsf{T}
$$

对于普通训练来说，到这里已经拿到了所有参数梯度，可以更新 $W_1,b_1,W_2,b_2,W_3,b_3$。

### 10.3 上一层输出变化由局部梯度确定

以：

$$
Z_3=A_2W_3+b_3
$$

为例，$A_2$ 是上一层输出。它通过 $W_3$ 影响 $Z_3$，又通过 $Z_3$ 影响 loss。

链式法则说：

$$
\frac{\partial L}{\partial A_2}
=
\frac{\partial L}{\partial Z_3}
\frac{\partial Z_3}{\partial A_2}
$$

矩阵形式就是：

$$
\frac{\partial L}{\partial A_2}
=
G_3W_3^\mathsf{T}
$$

所以不是我们“猜”上一层应该怎么变，而是根据后一层的权重和后一层传回来的梯度算出来。

直觉上：

- 如果某个隐藏特征通过正权重增加了错误类别的 logit，它会收到“往下调”的梯度；
- 如果某个隐藏特征通过正权重增加了正确类别的 logit，它会收到“往上调”的梯度；
- 如果它前面的 ReLU 是关闭的，也就是 $Z\le0$，那梯度会被 ReLU 截断为 0。

### 10.4 多层 MLP 的 NumPy 实现骨架

```python
class MLP:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
```

如果 `layers` 是：

```python
layers = [
    Linear(2, 16, rng),
    ReLU(),
    Linear(16, 16, rng),
    ReLU(),
    Linear(16, 2, rng),
]
```

那么前向传播就是按列表从前往后执行，反向传播就是按列表从后往前执行。

这就是多层 MLP 代码的核心框架。

### 10.5 前向缓存支持逐层反向传播

“从后往前”不代表要重新做一次前向计算。前向经过每一层时，要把反向需要的信息保存下来：

- Linear 保存它当时收到的输入；计算权重梯度时需要它。
- ReLU 保存哪些位置大于 0；反向时只有这些位置允许梯度通过。
- 每层自己的权重会保留在层对象中；计算“传给前一层的反馈”时需要它。

因此，`cache` 不是额外的神秘数据，而是“这层刚才到底看到了什么”的小笔记。下面是把整条公式流程翻译成 NumPy 伪代码的版本：

```python
# 前向：从输入一路走到 logits，并保存每层需要的中间量
A = X
caches = []

for layer in hidden_layers:
    Z = A @ layer.W + layer.b
    A_next = relu(Z)
    caches.append((A, Z, layer.W))
    A = A_next

logits = A @ W_out + b_out
loss = cross_entropy(logits, y)

# 反向：从输出层的错误开始，按相反顺序把反馈传回去
G = d_cross_entropy_d_logits(logits, y)

grad_W_out = A.T @ G
grad_b_out = G.sum(axis=0)
G = G @ W_out.T

for A_prev, Z, W in reversed(caches):
    G = G * (Z > 0)       # 先穿过 ReLU
    grad_W = A_prev.T @ G # 再得到本层参数梯度
    grad_b = G.sum(axis=0)
    G = G @ W.T           # 继续传给更前一层
```

每一层重复同一个规则：接收后一层传来的反馈，算自己的参数该如何改，再把“本层输入应该如何改”的反馈传给前一层。这就是多层反向传播。

---

## 11. MLP 的表达能力与决策边界

一个 MLP（多层感知机）可以写为：

$$
Z_1=XW_1+b_1,\quad H=ReLU(Z_1),\quad Z_2=HW_2+b_2
$$

`Z2` 是 logits。线性层叠线性层仍然只是线性变换；ReLU 提供非线性，使网络能表达弯曲的决策边界。

$$
ReLU(x)=\max(0,x)
$$

## 12. NumPy MLP 的核心组件实现

本节仍然按组件讲解，方便逐个理解和调试；完整拼装后的可运行脚本见：[完整模型代码](02_NumPy神经网络与PyTorch_完整模型代码.md)。

### 12.1 Linear 层：线性变换与参数梯度

```python
import numpy as np

class Linear:
    def __init__(self, in_features, out_features, rng):
        # 简单的方差缩放初始化，避免激活一开始过大
        self.W = rng.normal(0, np.sqrt(2 / in_features),
                            size=(in_features, out_features))
        self.b = np.zeros(out_features)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        self.grad_W = self.x.T @ grad_out
        self.grad_b = grad_out.sum(axis=0)
        return grad_out @ self.W.T
```

形状检查：`x:[B,in]`，`W:[in,out]`，`grad_out:[B,out]`，所以 `grad_W:[in,out]`，传回输入的梯度为 `[B,in]`。

### 12.2 ReLU 层：非线性与梯度掩码

```python
class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, grad_out):
        return grad_out * self.mask
```

### 12.3 MLP 容器：按顺序组织网络层

有了 `Linear` 和 `ReLU`，多层 MLP 不需要为每一层写一套新逻辑，只要把层放进列表：

```python
class MLP:
    def __init__(self, input_dim, hidden_dims, num_classes, rng):
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(Linear(dims[i], dims[i + 1], rng))

            # 最后一层输出 logits，不接 ReLU
            if i < len(dims) - 2:
                layers.append(ReLU())

        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
```

如果写：

```python
model = MLP(
    input_dim=2,
    hidden_dims=[32, 32],
    num_classes=2,
    rng=rng,
)
```

它实际表示：

```text
Linear(2, 32)
ReLU()
Linear(32, 32)
ReLU()
Linear(32, 2)
```

前向传播按顺序执行：

```text
第 1 层 → 第 2 层 → 第 3 层 → ...
```

反向传播按反方向执行：

```text
最后一层 → 倒数第二层 → ... → 第一层
```

这就是多层神经网络的程序框架：每一层只负责自己的 forward/backward，`MLP` 容器负责把层串起来。

这段代码是在把多个 `Linear` 和 `ReLU` 层自动组装成一个多层感知机 MLP，并规定前向传播和反向传播如何依次穿过这些层。

---

**MLP 类的职责**

```python
class MLP:
```

`MLP` 是多层感知机，英文是 Multi-Layer Perceptron。

它通常由若干个这样的结构组成：

```
Linear → ReLU → Linear → ReLU → Linear
```

最后一层一般只输出 logits，不接 ReLU。

---

**初始化网络结构**

```python
def __init__(self, input_dim, hidden_dims, num_classes, rng):
```

参数含义：

- `input_dim`：输入特征数
- `hidden_dims`：隐藏层维度
- `num_classes`：类别数，也就是最终输出维度
- `rng`：随机数生成器，用来初始化权重

例如：

```python
model = MLP(
    input_dim=4,
    hidden_dims=[8, 6],
    num_classes=3,
    rng=rng
)
```

表示网络结构是：

```
4 → 8 → 6 → 3
```

也就是：

```
输入层：4 个特征
第一隐藏层：8 个神经元
第二隐藏层：6 个神经元
输出层：3 个类别
```

---

**构造各层维度列表**

```python
dims = [input_dim] + list(hidden_dims) + [num_classes]
```

假设：

```python
input_dim = 4
hidden_dims = [8, 6]
num_classes = 3
```

那么：

```python
[input_dim]
```

是：

```python
[4]
```

```python
list(hidden_dims)
```

是：

```python
[8, 6]
```

```python
[num_classes]
```

是：

```python
[3]
```

拼起来：

```python
dims = [4] + [8, 6] + [3]
```

得到：

```python
dims = [4, 8, 6, 3]
```

这串数字描述了每一层的维度。

---

**创建网络层容器**

```python
layers = []
```

先创建一个空列表，后面逐个把层加入进去。

最终可能变成：

```python
[
    Linear(4, 8),
    ReLU(),
    Linear(8, 6),
    ReLU(),
    Linear(6, 3)
]
```

---

**循环构造相邻线性层**

```python
for i in range(len(dims) - 1):
```

假设：

```python
dims = [4, 8, 6, 3]
```

那么：

```python
len(dims)
```

是 `4`，所以：

```python
range(len(dims) - 1)
```

就是：

```python
range(3)
```

`i` 会依次取：

```
0、1、2
```

为什么是 `len(dims) - 1`？

因为四个维度之间只有三次连接：

```
4 → 8
8 → 6
6 → 3
```

所以需要三个 `Linear` 层。

---

**添加 Linear 层并检查形状**

```python
layers.append(Linear(dims[i], dims[i + 1], rng))
```

每次用相邻两个维度创建一个线性层。

**第一次构建：输入层到第一隐藏层**

```python
i = 0
```

所以：

```python
Linear(dims[0], dims[1], rng)
```

即：

```python
Linear(4, 8, rng)
```

**第二次构建：第一隐藏层到第二隐藏层**

```python
i = 1
```

得到：

```python
Linear(8, 6, rng)
```

**第三次构建：第二隐藏层到输出层**

```python
i = 2
```

得到：

```python
Linear(6, 3, rng)
```

---

**输出层不接 ReLU 的原因**

```python
if i < len(dims) - 2:
    layers.append(ReLU())
```

还是假设：

```python
dims = [4, 8, 6, 3]
```

那么：

```python
len(dims) - 2 = 2
```

所以判断条件是：

```python
if i < 2:
```

**索引 `i = 0` 时的层结构**

```python
0 < 2
```

成立，添加 ReLU。

**索引 `i = 1` 时的层结构**

```python
1 < 2
```

成立，添加 ReLU。

**索引 `i = 2` 时的层结构**

```python
2 < 2
```

不成立，不添加 ReLU。

所以最终结构是：

```
Linear(4,8)
ReLU
Linear(8,6)
ReLU
Linear(6,3)
```

最后一层输出的是 logits。

如果最后一层也接 ReLU，那么 logits 会全部变成非负数，这会不必要地限制模型输出。

分类任务中通常直接把最后的 logits 交给交叉熵：

```python
loss = cross_entropy(logits, labels)
```

---

**保存构建完成的网络层**

```python
self.layers = layers
```

把局部变量 `layers` 保存到当前模型对象中。

以后可以访问：

```python
model.layers
```

如果不写 `self.layers`，初始化函数结束后，局部变量 `layers` 就不能在其他方法中使用了。

---

**MLP 容器的顺序前向传播**

```python
def forward(self, x):
    for layer in self.layers:
        x = layer.forward(x)
    return x
```

这段代码让输入按顺序穿过所有层。

假设：

```python
self.layers = [
    Linear1,
    ReLU1,
    Linear2,
    ReLU2,
    Linear3
]
```

循环等价于：

```python
x = Linear1.forward(x)
x = ReLU1.forward(x)
x = Linear2.forward(x)
x = ReLU2.forward(x)
x = Linear3.forward(x)
```

最后：

```python
return x
```

返回最终 logits。

---

**逐层覆盖 `x` 的原因**

```python
x = layer.forward(x)
```

因为每一层的输出，要成为下一层的输入。

流程是：

```
原始 x
  ↓ Linear1
新的 x
  ↓ ReLU1
新的 x
  ↓ Linear2
新的 x
  ↓ ReLU2
新的 x
  ↓ Linear3
最终 logits
```

这里变量名一直叫 `x`，但其中保存的内容不断变化。

---

**MLP 容器的逆序反向传播**

```python
def backward(self, grad):
    for layer in reversed(self.layers):
        grad = layer.backward(grad)
    return grad
```

前向传播是正序：

```
Linear1 → ReLU1 → Linear2 → ReLU2 → Linear3
```

反向传播必须倒序：

```
Linear3 → ReLU2 → Linear2 → ReLU1 → Linear1
```

因此使用：

```python
reversed(self.layers)
```

---

**反向传播必须逆序执行的原因**

因为复合函数求导遵循链式法则。

假设：

```
x → layer1 → h1 → layer2 → h2 → loss
```

求梯度时要从损失开始：

```
∂L/∂h2
   ↓
layer2.backward
   ↓
∂L/∂h1
   ↓
layer1.backward
   ↓
∂L/∂x
```

不能从第一层开始，因为第一层要计算梯度时，需要先知道后一层传回来的梯度。

---

**`grad = layer.backward(grad)` 的数据含义**

最开始的 `grad` 通常是：

$$
\frac{\partial L}{\partial \text{logits}}
$$

它先传给最后一层：

```python
grad = last_layer.backward(grad)
```

最后一层返回：

$$
\frac{\partial L}{\partial \text{它的输入}}
$$

这个返回值再传给前一层：

```python
grad = previous_layer.backward(grad)
```

所以每一次循环，`grad` 的含义都在变化。

例如：

```
最开始：
grad = ∂L/∂Z3

经过最后一个 Linear：
grad = ∂L/∂A2

经过 ReLU：
grad = ∂L/∂Z2

经过前一个 Linear：
grad = ∂L/∂A1
```

---

**两层隐藏层的完整 MLP 示例**

```python
model = MLP(
    input_dim=4,
    hidden_dims=[8, 6],
    num_classes=3,
    rng=rng
)
```

构造出的层：

```
self.layers[0] = Linear(4, 8)
self.layers[1] = ReLU()
self.layers[2] = Linear(8, 6)
self.layers[3] = ReLU()
self.layers[4] = Linear(6, 3)
```

前向：

```
X: (batch, 4)
 ↓ Linear(4,8)
(batch, 8)
 ↓ ReLU
(batch, 8)
 ↓ Linear(8,6)
(batch, 6)
 ↓ ReLU
(batch, 6)
 ↓ Linear(6,3)
logits: (batch, 3)
```

反向：

```
∂L/∂logits: (batch, 3)
 ↓ Linear(6,3).backward
(batch, 6)
 ↓ ReLU.backward
(batch, 6)
 ↓ Linear(8,6).backward
(batch, 8)
 ↓ ReLU.backward
(batch, 8)
 ↓ Linear(4,8).backward
(batch, 4)
```

---

**MLP 容器实现的核心要点**

前向传播：

```python
for layer in self.layers:
```

按正常顺序经过每一层。

反向传播：

```python
for layer in reversed(self.layers):
```

按相反顺序经过每一层。

整个类的核心就是：

```
初始化：自动搭建网络
forward：从前往后计算
backward：从后往前传梯度
```

你问的其实是两个关键点：

1. `CrossEntropy` 到底是什么；
2. `MLP.forward()` 看起来没有保存中间结果，反向传播怎么还能算。

---

### 12.4 Softmax 与交叉熵的整体数据流

`CrossEntropy` 中文叫**交叉熵损失**，主要用于分类任务。

假设有三个类别，模型最后输出：

```
logits = [2.0, 1.0, 0.1]
```

这些不是概率，只是模型对三个类别的原始评分。

如果真实类别是第 0 类，可以写成 one-hot：

```
y = [1, 0, 0]
```

先经过 Softmax 得到概率：

```
p ≈ [0.659, 0.242, 0.099]
```

交叉熵损失是：

$$
L=-\sum_j y_j\log p_j
$$

因为真实标签中只有正确类别位置是 1，所以实际相当于：

$$
L=-\log p_{\text{正确类别}}
$$

这里正确类别概率是 `0.659`，所以：

$$
L=-\log 0.659
$$

大约是：

```
0.417
```

**交叉熵衡量的对象**

交叉熵在惩罚：

> 模型给正确类别的概率太低。

例如正确类别概率：

```
0.9  → 损失很小
0.5  → 损失较大
0.01 → 损失非常大
```

所以训练目标就是让正确类别的概率越来越高。

---

**交叉熵不需要预先执行 Softmax 的原因**

代码一般写成：

```python
loss, grad = cross_entropy(logits, labels)
```

而不是：

```python
probs = softmax(logits)
loss, grad = cross_entropy(probs, labels)
```

因为交叉熵函数通常内部已经完成：

```
logits
  ↓
softmax / log-softmax
  ↓
计算损失
```

并且它还能直接得到对 logits 的梯度：

$$
\frac{\partial L}{\partial Z}
=
p-y
$$

其中：

- $Z$：logits
- $p$：Softmax 概率
- $y$：真实标签 one-hot

例如：

```
p = [0.659, 0.242, 0.099]
y = [1,     0,     0]
```

梯度就是：

```
grad = [-0.341, 0.242, 0.099]
```

这个梯度会成为整个神经网络反向传播的起点：

```python
model.backward(grad)
```

---

**各层对象独立管理前向缓存**

代码是：

```python
def forward(self, x):
    for layer in self.layers:
        x = layer.forward(x)
    return x
```

看起来只是让 `x` 一路向后走，并没有写：

```python
cache.append(x)
```

但缓存并不是保存在 `MLP` 里，而是保存在**每个层对象自己内部**。

例如你之前的 `Linear`：

```python
def forward(self, x):
    self.x = x
    return x @ self.W + self.b
```

这里：

```python
self.x = x
```

就是在缓存中间数据。

每个 `Linear` 实例都有自己的 `self.x`。

---

**各 Linear 层独立保存输入缓存**

假设网络是：

```
Linear1 → ReLU1 → Linear2 → ReLU2 → Linear3
```

前向传播时：

```python
x = Linear1.forward(x)
```

`Linear1` 内部保存：

```python
Linear1.x = 原始输入
```

然后：

```python
x = ReLU1.forward(x)
```

`ReLU1` 内部可能保存：

```python
ReLU1.mask = 输入是否大于 0
```

然后：

```python
x = Linear2.forward(x)
```

`Linear2` 内部保存：

```python
Linear2.x = ReLU1 的输出
```

虽然这些对象都属于 `Linear` 类，但它们是不同实例，所以各自有独立的数据：

```
Linear1.x
Linear2.x
Linear3.x
```

互不覆盖。

---

**逐层前向传播的数据变化**

假设原始输入是 $X$。

第一层：

$$
Z_1=XW_1+b_1
$$

此时第一个 Linear 保存：

```
Linear1.x = X
```

经过 ReLU：

$$
A_1=\operatorname{ReLU}(Z_1)
$$

ReLU 保存：

```
ReLU1.mask = Z1 > 0
```

第二个 Linear：

$$
Z_2=A_1W_2+b_2
$$

第二个 Linear 保存：

```
Linear2.x = A1
```

所以虽然 MLP 没有统一建立一个大缓存，每一层都保存了自己反向传播所需的信息。

---

**ReLU 层需要保存的缓存**

一个可能的 `ReLU` 实现是：

```python
class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, grad_out):
        return grad_out * self.mask
```

前向传播时：

```python
self.mask = x > 0
```

记录哪些位置原来是正数。

例如输入：

```
[-2, 3, 0, 5]
```

mask 是：

```
[False, True, False, True]
```

反向传播时：

```python
grad_out * self.mask
```

负数位置的梯度变成 0，正数位置保留。

因此 ReLU 不一定要保存完整的 `x`，只保存一个布尔掩码就够了。

---

**反向传播对各层缓存的使用**

MLP 的反向传播是：

```python
def backward(self, grad):
    for layer in reversed(self.layers):
        grad = layer.backward(grad)
    return grad
```

它反向经过所有层。

假设正向是：

```
Linear1 → ReLU1 → Linear2 → ReLU2 → Linear3
```

反向就是：

```
Linear3 → ReLU2 → Linear2 → ReLU1 → Linear1
```

当走到 `Linear3.backward()` 时，它使用自己前向保存的：

```python
Linear3.x
```

计算：

```python
self.grad_W = self.x.T @ grad_out
self.grad_b = grad_out.sum(axis=0)
grad_x = grad_out @ self.W.T
```

走到 `ReLU2.backward()` 时，它使用自己保存的：

```python
ReLU2.mask
```

走到 `Linear2.backward()` 时，它使用：

```python
Linear2.x
```

所以每层都知道自己该怎么反向计算。

---

**中间激活值与模型参数的更新方式**

这里要区分两种东西。

**中间激活值在前向传播中重新计算**

例如：

```
Z1、A1、Z2、A2
```

它们通常不是被梯度下降“更新”的参数。

每一次前向传播，都会根据当前输入和参数重新计算，并覆盖旧缓存：

```python
self.x = x
```

它们只是临时数据。

**模型参数由优化器更新**

真正需要训练更新的是：

```
W、b
```

`backward()` 只负责算出梯度：

```python
self.grad_W
self.grad_b
```

还需要优化器或手动代码更新：

```python
layer.W -= lr * layer.grad_W
layer.b -= lr * layer.grad_b
```

所以完整流程是：

```python
#### 1. 前向传播，并由各层缓存中间数据
logits = model.forward(x)

#### 2. 交叉熵计算损失和对 logits 的梯度
loss, grad_logits = cross_entropy(logits, labels)

#### 3. 反向传播，各层利用自己的缓存计算梯度
model.backward(grad_logits)

#### 4. 更新各个 Linear 层的 W 和 b
for layer in model.layers:
    if isinstance(layer, Linear):
        layer.W -= lr * layer.grad_W
        layer.b -= lr * layer.grad_b
```

---

**MLP 容器不集中保存中间值的原因**

当然也可以集中保存：

```python
activations = [x]
```

但面向对象的写法通常让每个层自己负责：

```
Linear 保存输入 x
ReLU 保存 mask
其他层保存自己需要的信息
```

这样 `MLP` 只负责安排顺序：

```
谁先执行
谁后执行
反向时谁先回来
```

而每个层负责自己的数学计算和缓存。

可以理解为：

> MLP 是总调度员，每个 layer 是独立工位。工位自己保存工作记录，反向检查时再拿出来用。

---

**缓存机制的使用限制**

因为缓存保存在：

```python
self.x
self.mask
```

所以一般必须按顺序：

```
forward → backward
```

如果你连续做两次前向：

```python
model.forward(x1)
model.forward(x2)
model.backward(grad)
```

第一次 `x1` 的缓存会被第二次 `x2` 覆盖，反向传播使用的会是 `x2` 的缓存。

这种简化实现适合：

```
一次 forward
一次 backward
再更新参数
```

真正的 PyTorch 会建立计算图，能管理更复杂的情形。

最核心的一句话是：

> `MLP.forward()` 本身没有集中缓存，但每个 `Linear`、`ReLU` 对象在自己的 `forward()` 中保存了反向传播所需的信息；`backward()` 倒序调用它们时，各层会取出自己的缓存计算梯度。

### 12.5 SoftmaxCrossEntropy 的稳定实现

把二者合并可得到更稳定、简洁的梯度：

```python
class SoftmaxCrossEntropy:
    def forward(self, logits, targets):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / exp.sum(axis=1, keepdims=True)
        self.targets = targets
        n = logits.shape[0]
        return -np.log(self.probs[np.arange(n), targets] + 1e-12).mean()

    def backward(self):
        n = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(n), self.targets] -= 1
        return grad / n
```

为什么梯度是 `p-y_onehot`？Softmax 的各类别互相耦合，与交叉熵求导后恰好化简：错误类别按当前概率向下压，正确类别增加 `-1` 的纠正项。

这段代码是在手写一个**“Softmax + 交叉熵损失”合并层**。

它做两件事：

```
forward：根据 logits 和真实标签，算出一个损失值
backward：算出损失对 logits 的梯度
```

最关键的结果是：

$$
\frac{\partial L}{\partial logits}
=
\frac{p-y_{\text{one-hot}}}{n}
$$

也就是图片最下面写的：

```
p - y_onehot
```

---

**输入的 Logits 与真实标签**

假设一个 batch 有 3 个样本，要分成 4 类：

```python
logits.shape == (3, 4)
```

例如：

```python
logits = np.array([
    [2.0, 1.0, 0.1, -1.0],
    [0.2, 1.5, 0.4, 0.8],
    [1.0, 0.5, 2.0, 0.1]
])
```

每一行对应一个样本，每一列对应一个类别。

真实类别可能是：

```python
targets = np.array([0, 3, 2])
```

意思是：

- 第 1 个样本正确类别是第 0 类；
- 第 2 个样本正确类别是第 3 类；
- 第 3 个样本正确类别是第 2 类。

---

**SoftmaxCrossEntropy 的前向计算**

```python
def forward(self, logits, targets):
```

接收：

- `logits`：模型原始输出；
- `targets`：每个样本的正确类别编号。

---

**稳定计算 Softmax 概率**

```python
shifted = logits - logits.max(axis=1, keepdims=True)
```

每一行减去这一行的最大值。

例如：

```
[2.0, 1.0, 0.1]
```

减去最大值 `2.0`：

```
[0.0, -1.0, -1.9]
```

这样是为了防止：

```python
np.exp(logits)
```

出现指数溢出。

`axis=1` 表示每一行分别处理。

---

```python
exp = np.exp(shifted)
```

对每个数取指数。

---

```python
self.probs = exp / exp.sum(axis=1, keepdims=True)
```

每一行除以这一行指数的总和，得到 Softmax 概率。

例如：

```
logits = [2.0, 1.0, 0.1]
```

可能得到：

```
probs ≈ [0.659, 0.242, 0.099]
```

并把它保存到：

```python
self.probs
```

因为反向传播还要用。

---

**保存真实标签用于反向传播**

```python
self.targets = targets
```

把正确类别也缓存起来，反向传播需要知道哪个位置是正确类别。

---

**读取 Batch 大小**

```python
n = logits.shape[0]
```

例如：

```python
logits.shape == (32, 10)
```

那么：

```python
n = 32
```

代表有 32 个样本。

---

**提取每个样本的正确类别概率**

最难的是这一句：

```python
self.probs[np.arange(n), targets]
```

假设：

```python
targets = [0, 2, 1]
```

那么：

```python
np.arange(n)
```

是：

```
[0, 1, 2]
```

组合起来取的是：

```
self.probs[0, 0]
self.probs[1, 2]
self.probs[2, 1]
```

也就是：

> 对第 0 个样本，取第 0 类的概率；
> 对第 1 个样本，取第 2 类的概率；
> 对第 2 个样本，取第 1 类的概率。

这些正好都是每个样本的**正确类别概率**。

例如：

```python
probs = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.3, 0.6],
    [0.2, 0.5, 0.3]
])

targets = np.array([0, 2, 1])
```

那么：

```python
probs[np.arange(3), targets]
```

得到：

```
[0.7, 0.6, 0.5]
```

---

**计算批量平均交叉熵**

```python
return -np.log(
    self.probs[np.arange(n), targets] + 1e-12
).mean()
```

交叉熵对每个样本做：

$$
L_n=-\log p_{\text{正确类别}}
$$

如果正确类别概率很高：

```
p = 0.9
```

那么损失很小：

$$
-\log 0.9\approx 0.105
$$

如果正确类别概率很低：

```
p = 0.01
```

那么损失很大：

$$
-\log 0.01\approx 4.605
$$

最后：

```python
.mean()
```

对 batch 中所有样本的损失取平均。

---

**`1e-12` 防止对零取对数**

```python
+ 1e-12
```

其中：

```
1e-12 = 0.000000000001
```

它是一个非常小的数，用来避免：

```python
np.log(0)
```

因为：

$$
\log 0=-\infty
$$

会导致数值问题。

---

**SoftmaxCrossEntropy 的反向计算**

```python
def backward(self):
```

它要计算：

$$
\frac{\partial L}{\partial logits}
$$

这是传给最后一个 `Linear` 层的上游梯度。

---

**读取 Batch 大小**

```python
n = self.probs.shape[0]
```

与前面的 `n` 含义相同。

---

**复制概率矩阵以避免破坏缓存**

```python
grad = self.probs.copy()
```

先让梯度等于 Softmax 概率：

```
grad = p
```

为什么用 `.copy()`？

因为后面要修改 `grad`。如果直接写：

```python
grad = self.probs
```

那么修改 `grad` 也会修改 `self.probs`，因为两者指向同一个数组。

---

**在正确类别位置减去 1**

```python
grad[np.arange(n), self.targets] -= 1
```

这就是把：

```
p
```

变成：

```
p - y_onehot
```

举个例子。

模型概率：

```
p = [0.7, 0.2, 0.1]
```

真实类别是第 0 类，对应 one-hot：

```
y = [1, 0, 0]
```

那么：

```
p - y
=
[0.7, 0.2, 0.1] - [1, 0, 0]
=
[-0.3, 0.2, 0.1]
```

代码没有真的构造：

```python
[1, 0, 0]
```

而是直接在正确类别位置减 `1`：

```python
grad[正确类别] -= 1
```

这样更节省内存。

---

**按 Batch 大小平均梯度**

```python
return grad / n
```

因为前向损失用了：

```python
.mean()
```

也就是对 `n` 个样本取平均。

所以反向梯度也要除以 `n`：

$$
\frac{\partial L}{\partial logits}
=
\frac{p-y}{n}
$$

如果前向使用的是求和：

```python
.sum()
```

而不是 `.mean()`，这里通常就不需要除以 `n`。

---

**梯度化简为 $p-y$ 的原因**

先看直观含义。

假设真实类别是第 0 类：

```
y = [1, 0, 0]
```

模型概率：

```
p = [0.7, 0.2, 0.1]
```

那么：

```
p - y = [-0.3, 0.2, 0.1]
```

梯度下降的更新方向是：

```
参数 -= 学习率 × 梯度
```

对正确类别，梯度是负数：

```
-0.3
```

减去负数，相当于增加正确类别的 logit。

对错误类别，梯度是正数：

```
0.2, 0.1
```

减去正数，会降低错误类别的 logits。

所以它的作用是：

```
正确类别：往上推
错误类别：往下压
```

而且错误类别当前概率越高，受到的惩罚就越大。

---

**Softmax 与交叉熵合并实现的原因**

理论上可以拆成两层：

```
logits
  ↓ Softmax
probs
  ↓ CrossEntropy
loss
```

反向时还要分别求：

```
CrossEntropy.backward
Softmax.backward
```

但是两者连在一起求导后，会大幅简化为：

$$
\frac{\partial L}{\partial logits}=p-y
$$

因此可以直接写成一个类：

```python
SoftmaxCrossEntropy
```

好处是：

- 公式更简单；
- 计算更快；
- 不需要显式构造 Softmax 的雅可比矩阵；
- 数值通常更稳定。

---

**SoftmaxCrossEntropy 与 MLP 的连接方式**

完整训练过程大致是：

```python
#### 1. 模型前向传播，得到 logits
logits = model.forward(x)

#### 2. 交叉熵前向传播，得到损失
loss = criterion.forward(logits, targets)

#### 3. 得到损失对 logits 的梯度
grad_logits = criterion.backward()

#### 4. 把梯度传回整个 MLP
model.backward(grad_logits)
```

其中：

```python
grad_logits
```

就是你前面一直看到的：

$$
G=\frac{\partial L}{\partial Z}
$$

最后一个 `Linear` 层接收到它：

```python
grad = last_linear.backward(grad_logits)
```

然后梯度再一层层向前传。

---

**损失层与 MLP 的连接结构**

```
输入 X
  ↓
MLP.forward
  ↓
logits
  ↓
SoftmaxCrossEntropy.forward
  ↓
loss
```

反向：

```
loss
  ↓
SoftmaxCrossEntropy.backward
  ↓
grad_logits = (p - y) / n
  ↓
MLP.backward
  ↓
各个 Linear 得到 grad_W、grad_b
```

所以这段代码本质上就是：

> 在前向传播中计算分类损失，在反向传播中给出神经网络开始反传所需的第一个梯度。

### 12.6 SGD 参数更新

```python
def sgd_step(layers, lr):
    for layer in layers:
        if hasattr(layer, "W"):
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b
```

这段代码是在给你前面手写的 NumPy 神经网络做一次最基础的 **SGD 参数更新**。

```python
def sgd_step(layers, lr):
    for layer in layers:
        if hasattr(layer, "W"):
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b
```

先说整体意思：

> 遍历网络中的所有层，只要这一层有权重 `W`，就按照梯度下降公式更新它的 `W` 和 `b`。

---

**`sgd_step` 函数的输入**

定义一个函数：

```python
sgd_step(layers, lr)
```

两个参数：

- `layers`：模型中所有层的列表
- `lr`：学习率 learning rate

例如：

```python
model.layers
```

可能是：

```python
[
    Linear(...),
    ReLU(),
    Linear(...),
    ReLU(),
    Linear(...)
]
```

调用：

```python
sgd_step(model.layers, lr=0.01)
```

就是让所有可训练层更新一次参数。

---

**逐层遍历网络参数**

```python
for layer in layers:
```

逐个取出网络中的层。

例如第一次循环：

```python
layer = 第一个 Linear 对象
```

第二次：

```python
layer = 第一个 ReLU 对象
```

第三次：

```python
layer = 第二个 Linear 对象
```

依次检查所有层。

---

**使用 `hasattr` 筛选参数层的原因**

```python
if hasattr(layer, "W"):
```

`hasattr()` 的意思是：

> 检查这个对象是否拥有某个属性。

这里检查当前层有没有：

```python
layer.W
```

你的 `Linear` 类有：

```python
self.W
self.b
```

所以：

```python
hasattr(linear_layer, "W")
```

结果是：

```python
True
```

而 `ReLU` 没有权重：

```python
hasattr(relu_layer, "W")
```

结果是：

```python
False
```

所以这句实际上是在筛选：

> 只更新 Linear 层，跳过 ReLU 层。

因为 ReLU 没有可以训练的参数。

---

**根据梯度更新权重**

```python
layer.W -= lr * layer.grad_W
```

它等价于：

```python
layer.W = layer.W - lr * layer.grad_W
```

数学上是：

$$
W_{\text{new}}
=
W_{\text{old}}
-
\eta \frac{\partial L}{\partial W}
$$

其中：

- $W$：权重
- $\frac{\partial L}{\partial W}$：损失对权重的梯度
- $\eta$：学习率，也就是 `lr`

你之前在反向传播中已经算出了：

```python
self.grad_W = self.x.T @ grad_out
```

所以 `sgd_step()` 只是拿这个梯度来更新权重。

---

**根据梯度更新偏置**

```python
layer.b -= lr * layer.grad_b
```

等价于：

```python
layer.b = layer.b - lr * layer.grad_b
```

数学公式：

$$
b_{\text{new}}
=
b_{\text{old}}
-
\eta \frac{\partial L}{\partial b}
$$

其中 `grad_b` 是之前反向传播得到的：

```python
self.grad_b = grad_out.sum(axis=0)
```

---

**沿负梯度方向更新的原因**

梯度指向函数增长最快的方向。

如果想让损失 $L$ 变小，就要沿梯度的反方向走：

$$
参数 \leftarrow 参数-\text{学习率}\times梯度
$$

例如：

$$
L(w)=w^2
$$

当：

$$
w=3
$$

梯度为：

$$
\frac{dL}{dw}=2w=6
$$

若学习率：

$$
lr=0.1
$$

更新：

$$
w_{\text{new}}
=
3-0.1\times6
=
2.4
$$

损失从：

$$
3^2=9
$$

变为：

$$
2.4^2=5.76
$$

确实下降了。

---

**SGD 数组更新示例**

假设某层当前：

```python
layer.W = np.array([
    [1.0, 2.0],
    [3.0, 4.0]
])

layer.grad_W = np.array([
    [0.2, -0.1],
    [0.5, 0.3]
])

lr = 0.1
```

更新：

```python
layer.W -= lr * layer.grad_W
```

先算：

```
lr * grad_W
=
0.1 *
[[ 0.2, -0.1],
 [ 0.5,  0.3]]
=
[[ 0.02, -0.01],
 [ 0.05,  0.03]]
```

然后：

```
新 W
=
[[1.0, 2.0],
 [3.0, 4.0]]
-
[[0.02, -0.01],
 [0.05,  0.03]]
```

得到：

```
[[0.98, 2.01],
 [2.95, 3.97]]
```

正梯度位置会减小，负梯度位置会增大。

---

**SGD 与 `backward()` 的职责分工**

`backward()` 做的是：

```
计算梯度
```

例如：

```python
layer.grad_W
layer.grad_b
```

而 `sgd_step()` 做的是：

```
使用梯度修改参数
```

所以两者不能混淆。

流程是：

```python
model.forward(x)       # 算预测
loss.forward(...)      # 算损失
model.backward(...)    # 算 grad_W、grad_b
sgd_step(...)          # 真正修改 W、b
```

---

**先计算梯度再更新参数的原因**

假设你还没有执行：

```python
model.backward(grad)
```

那么：

```python
layer.grad_W
layer.grad_b
```

可能还不存在，或者还是上一轮留下的旧梯度。

因此必须先：

```python
model.backward(...)
```

再：

```python
sgd_step(...)
```

正确顺序：

```
前向
→ 损失
→ 反向算梯度
→ SGD 更新参数
```

---

**参数更新后必须重新前向传播的原因**

前向传播时，各层保存了缓存，例如：

```python
self.x
self.mask
```

这些缓存对应的是更新前的参数。

做完：

```python
layer.W -= ...
layer.b -= ...
```

参数已经变了。

所以旧缓存只属于上一次计算，下一轮必须重新执行：

```python
model.forward(new_x)
```

重新生成与新参数对应的中间结果。

---

**SGD 更新顺序总结**

```python
def sgd_step(layers, lr):
```

就是手写版的：

```python
optimizer.step()
```

它遍历所有层，找到有权重的 `Linear` 层，然后执行：

$$
W\leftarrow W-lr\cdot grad_W
$$

$$
b\leftarrow b-lr\cdot grad_b
$$

`backward()` 负责算梯度，`sgd_step()` 负责真正更新参数。

每轮顺序必须是：前向、计算损失、反向、更新。更新后旧计算图/缓存对应的已是旧参数。

## 13. 数值梯度检查

随机挑一个参数 `W[i,j]`，比较反向传播的解析梯度和中心差分数值梯度。相对误差：

```python
relative_error = abs(a - b) / max(1e-8, abs(a) + abs(b))
```

先用 `float64` 和很小的网络检查。若误差大，常见原因是漏除 batch size、转置错误、缓存被覆盖、ReLU 在 0 附近不可导。

## 14. NumPy MLP 的训练循环与故障诊断

最小训练循环：

```text
初始化参数
重复多个 epoch：
    打乱训练数据
    对每个 batch：
        logits = model(x)
        loss = criterion(logits, y)
        grads = backward(loss)
        update(parameters, grads)
    在验证集评估
保存验证指标最好的 checkpoint
```

诊断顺序：

1. 能否在 20 个样本上过拟合到接近 100%？不能则实现可能有 bug。
2. 损失是否有限且总体下降？
3. 随机标签是否无法泛化？若仍高分可能泄漏。
4. 零学习率时参数是否保持不变？
5. 保存再加载后，同一输入输出是否一致？

## 15. PyTorch 张量与自动微分概览

第 15～17 节先用最小代码快速建立 PyTorch 的整体数据流；第 18 节再把你补充的逐句笔记整合起来详细解释。建议先看一遍全局，再进入细节。

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print(x.grad)  # 2*x+3 = 7
```

PyTorch 在前向时记录计算图，`backward()` 用链式法则累积梯度。注意是“累积”：每个训练 step 前要清梯度。

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

评估时：

```python
model.eval()
with torch.no_grad():
    logits = model(x)
```

`eval()` 改变 Dropout/BatchNorm 等层的行为；`no_grad()` 停止记录梯度，二者解决不同问题。

## 16. PyTorch 数据集与批量加载

```python
from torch.utils.data import Dataset, DataLoader

class PairDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = torch.as_tensor(xs, dtype=torch.float32)
        self.ys = torch.as_tensor(ys, dtype=torch.long)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

loader = DataLoader(PairDataset(xs, ys), batch_size=32, shuffle=True)
```

分类目标通常用 `long`，特征通常用 `float32`。文本长度不同需要 padding 和 `collate_fn`；不能简单堆叠不同形状。

## 17. 使用 `nn.Module` 定义 MLP

```python
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, x):
        return self.net(x.flatten(1))
```

`CrossEntropyLoss` 需要原始 logits，不要先手工 Softmax；它内部进行了数值稳定的 `log_softmax + NLLLoss`。

## 18. PyTorch 核心组件的逐句解析

这张图在讲：**从“手写神经网络”切换到 PyTorch 的标准训练方式**。主要分成三块：

1. PyTorch 张量与自动微分
2. Dataset 与 DataLoader
3. 用 `nn.Module` 搭建 MLP

---

### 18.1 张量与自动微分

先看这段：

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x
y.backward()

print(x.grad)
```

它在计算：

$$
y=x^2+3x
$$

并自动求：

$$
\frac{dy}{dx}=2x+3
$$

因为：

```python
x = 2
```

所以：

$$
\frac{dy}{dx}=2\times 2+3=7
$$

因此：

```python
print(x.grad)
```

输出大致是：

```
tensor([7.])
```

**`requires_grad=True`：启用梯度跟踪**

```python
x = torch.tensor([2.0], requires_grad=True)
```

意思是：

> 告诉 PyTorch，需要追踪所有与 `x` 有关的运算，之后要对 `x` 求梯度。

如果不写：

```python
requires_grad=True
```

PyTorch 通常不会为这个张量建立求导所需的计算图。

---

**`y.backward()`：触发反向传播**

```python
y.backward()
```

意思是：

> 从 `y` 开始，按照链式法则向前反向求导，把梯度存进相关张量的 `.grad` 属性。

所以最后可以访问：

```python
x.grad
```

之前你手写网络时，是自己写：

```python
grad_W = x.T @ grad_out
grad_b = grad_out.sum(axis=0)
```

而 PyTorch 的自动微分会自动完成这些求导。

---

**梯度累积与参数更新顺序**

图中写了：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这是最标准的训练三步。

**清除上一轮累积梯度**

```python
optimizer.zero_grad()
```

PyTorch 默认会**累加梯度**，不是覆盖梯度。

例如第一次：

```
w.grad = 3
```

第二次再反向传播得到 `4`，如果没有清零，最后会是：

```
w.grad = 3 + 4 = 7
```

所以每一轮训练开始前通常要：

```python
optimizer.zero_grad()
```

把上一轮梯度清掉。

---

**根据当前损失执行反向传播**

```python
loss.backward()
```

自动计算损失对所有可训练参数的梯度：

```
W.grad
b.grad
...
```

---

**使用优化器更新参数**

```python
optimizer.step()
```

优化器根据梯度更新参数。

例如 SGD 大致相当于：

```python
W = W - lr * W.grad
b = b - lr * b.grad
```

所以完整顺序不能乱：

```python
optimizer.zero_grad()  # 清旧梯度
loss.backward()        # 算新梯度
optimizer.step()       # 更新参数
```

---

**训练模式与评估模式**

图片中还有：

```python
model.eval()

with torch.no_grad():
    logits = model(x)
```

这两句话解决的是两个不同问题。

**`model.eval()`：切换到评估模式**

```python
model.eval()
```

把模型切换到评估状态。

它主要会影响：

- `Dropout`
- `BatchNorm`

例如 Dropout 在训练时会随机丢弃部分神经元，但测试时不应该再随机丢弃，因此要调用：

```python
model.eval()
```

普通的 `Linear` 和 `ReLU` 在训练、评估模式下通常没有区别。

---

**`torch.no_grad()`：关闭梯度记录**

```python
with torch.no_grad():
    logits = model(x)
```

意思是：

> 这段代码只做前向计算，不建立计算图，也不准备反向传播。

因为测试时通常只需要预测，不需要求梯度。

优点是：

- 减少内存占用；
- 提高计算速度；
- 避免意外记录梯度。

所以：

```python
model.eval()
```

控制某些层的行为；

```python
torch.no_grad()
```

关闭自动求导记录。

二者不能互相替代。

---

### 18.2 Dataset 与 DataLoader

这部分是在解决：

> 数据怎么组织？怎么每次拿出一个 batch 训练？

代码大致是：

```python
from torch.utils.data import Dataset, DataLoader

class PairDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = torch.as_tensor(xs, dtype=torch.float32)
        self.ys = torch.as_tensor(ys, dtype=torch.long)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]
```

---

**`Dataset` 的数据索引职责**

`Dataset` 可以理解成：

> 一套规定数据如何存储、总共有多少条、如何取出第 `idx` 条数据的规则。

这里每条数据由一对内容组成：

```
特征 x
标签 y
```

所以叫：

```python
PairDataset
```

---

**`__init__`：保存特征与标签**

```python
def __init__(self, xs, ys):
    self.xs = torch.as_tensor(xs, dtype=torch.float32)
    self.ys = torch.as_tensor(ys, dtype=torch.long)
```

把输入数据转成 PyTorch 张量。

**特征张量通常使用浮点类型**

```python
dtype=torch.float32
```

因为神经网络里的矩阵乘法、权重等通常使用浮点数。

**分类标签通常使用整数类型**

```python
dtype=torch.long
```

例如：

```python
ys = [0, 2, 1, 0]
```

每个数字表示正确类别编号。

`CrossEntropyLoss` 通常要求类别标签是 `long` 类型。

---

**`__len__`：返回样本数量**

```python
def __len__(self):
    return len(self.ys)
```

规定：

```python
len(dataset)
```

应该返回多少。

假设有 1000 个标签，那么：

```python
len(dataset)
```

返回：

```
1000
```

---

**`__getitem__`：按索引返回单个样本**

```python
def __getitem__(self, idx):
    return self.xs[idx], self.ys[idx]
```

规定如何取出一条数据。

例如：

```python
dataset[5]
```

会返回：

```
第 5 个样本的特征
第 5 个样本的标签
```

即：

```python
(x, y)
```

---

**`DataLoader` 的批量迭代职责**

```python
loader = DataLoader(
    PairDataset(xs, ys),
    batch_size=32,
    shuffle=True
)
```

`DataLoader` 是负责从 Dataset 中批量拿数据的工具。

**`batch_size=32`：每批样本数量**

每次拿出 32 个样本：

```
第一个 batch：32 条
第二个 batch：32 条
第三个 batch：32 条
……
```

最后剩余不足 32 条时，通常会返回一个较小的 batch。

**`shuffle=True`：每轮打乱训练样本**

每一轮训练前打乱数据顺序。

避免模型总是按照完全相同的顺序看到数据。

训练时通常写：

```python
for xb, yb in loader:
    logits = model(xb)
    loss = criterion(logits, yb)
```

这里：

```
xb：一个 batch 的特征
yb：一个 batch 的标签
```

---

### 18.3 基于 `nn.Module` 的 MLP 定义

前面你手写的 MLP 是：

```
Linear
ReLU
Linear
ReLU
Linear
```

PyTorch 已经提供了现成版本：

```python
from torch import nn
```

模型一般这样写：

```python
class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, classes=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, classes)
        )

    def forward(self, x):
        return self.net(x)
```

---

**继承 `nn.Module` 的原因**

```python
class MLP(nn.Module):
```

表示这个类是一个 PyTorch 模型。

继承以后，PyTorch 才能自动管理：

- 模型参数；
- `.parameters()`；
- `.train()` 和 `.eval()`；
- 把模型放到 GPU；
- 保存和加载参数；
- 自动微分关联。

---

**`super().__init__()`：初始化父类能力**

```python
super().__init__()
```

调用父类 `nn.Module` 的初始化函数。

可以理解成：

> 先把 PyTorch 模型的基础功能初始化好，再创建自己的网络层。

漏掉这一句，参数注册等机制可能无法正常工作。

---

**`nn.Linear`：定义全连接层**

```python
nn.Linear(input_dim, hidden_dim)
```

就是你之前手写的：

$$
y=xW+b
$$

它内部自动创建：

```
weight
bias
```

并且自动参与梯度计算。

---

**`nn.Sequential`：按顺序组合网络层**

```python
self.net = nn.Sequential(
    nn.Linear(...),
    nn.ReLU(),
    nn.Linear(...)
)
```

作用是：

> 按顺序把各层串起来。

输入会自动执行：

```
x
↓ Linear
↓ ReLU
↓ Linear
输出
```

相当于你之前手写的：

```python
for layer in self.layers:
    x = layer.forward(x)
```

---

### 18.4 数据、模型、损失与自动微分的连接

完整训练代码大概是：

```python
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for xb, yb in loader:
    logits = model(xb)
    loss = criterion(logits, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

逐句对应：

```python
for xb, yb in loader:
```

DataLoader 提供一个 batch。

```python
logits = model(xb)
```

MLP 前向传播。

```python
loss = criterion(logits, yb)
```

交叉熵计算损失。

```python
optimizer.zero_grad()
```

清空旧梯度。

```python
loss.backward()
```

自动反向传播。

```python
optimizer.step()
```

更新权重和偏置。

整张图可以概括为：

> `Dataset/DataLoader` 负责喂数据，`nn.Module` 负责定义模型，自动微分负责求梯度，`optimizer` 负责更新参数。

## 19. PyTorch 完整训练与验证循环

```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.size(0)

    return {"loss": total_loss / total, "accuracy": total_correct / total}
```

乘 `y.size(0)` 是因为 `loss.item()` 通常是 batch 均值；最后除总样本数，避免最后一个小 batch 被赋予同样权重。

这张图主要讲两件事：

1. `train_one_epoch()`：模型怎样完整训练一轮；
2. checkpoint：怎样把训练进度保存下来。

---

### 19.1 Epoch 的整体训练结构

```python
def train_one_epoch(model, loader, optimizer, criterion, device):
```

`epoch` 是“把整个训练集完整学习一遍”。

假设：

```
训练集：1000 个样本
batch_size：32
```

那么一个 epoch 大约要处理 32 个 batch。

这个函数的五个参数分别是：

```python
model       # 模型
loader      # 分批提供训练数据
optimizer   # 更新模型参数
criterion   # 损失函数
device      # CPU 或 GPU
```

---

### 19.2 单个 Batch 的训练步骤

**切换到训练模式**

```python
model.train()
```

它告诉模型：

> 现在处于训练阶段。

它主要影响 `Dropout` 和 `BatchNorm`。

注意，它本身不会自动进行前向、反向和更新，只是切换模型状态。

---

**初始化 Epoch 统计变量**

```python
total_loss = 0.0
total_correct = 0
total = 0
```

用于统计整个 epoch 的表现：

- `total_loss`：所有样本的损失总和；
- `total_correct`：预测正确的样本总数；
- `total`：处理过的样本总数。

这些变量不参与模型训练，只是最后计算平均损失和准确率。

---

**逐 Batch 读取训练数据**

```python
for x, y in loader:
```

`loader` 每次给出一个 batch：

- `x`：这一批样本的特征；
- `y`：这一批样本的真实标签。

例如：

```
x.shape = (32, 784)
y.shape = (32,)
```

表示：

- 有 32 个样本；
- 每个样本有 784 个特征；
- 每个样本有一个类别编号。

循环会执行很多次，直到整个训练集都处理完。

---

**把数据移动到计算设备**

```python
x, y = x.to(device), y.to(device)
```

假设：

```python
device = "cuda"
```

那么就是把 `x` 和 `y` 移到 GPU。

模型和数据必须在同一个设备上，否则无法计算。

例如不能：

```
模型在 GPU
数据在 CPU
```

---

**清空优化器中的旧梯度**

```python
optimizer.zero_grad()
```

PyTorch 的梯度默认是累加的。

假设上一个 batch 得到：

```
W.grad = 2
```

当前 batch 又得到：

```
3
```

如果不清空，结果会变成：

```
W.grad = 2 + 3 = 5
```

普通训练中，我们希望每个 batch 独立计算梯度，所以要先：

```python
optimizer.zero_grad()
```

---

**调用模型完成前向传播**

```python
logits = model(x)
```

这相当于调用：

```python
logits = model.forward(x)
```

模型把输入一层层处理，最后得到 `logits`。

假设分类数量是 10：

```
x.shape      = (32, 784)
logits.shape = (32, 10)
```

每一行对应一个样本，每一列对应一个类别的原始评分。

---

**根据 Logits 与标签计算损失**

```python
loss = criterion(logits, y)
```

`criterion` 通常是：

```python
criterion = torch.nn.CrossEntropyLoss()
```

它根据：

- 模型预测的 `logits`
- 真实标签 `y`

计算当前 batch 的平均损失。

例如：

```
loss = tensor(0.7234)
```

这里的 `loss` 是一个只有一个数的张量。

---

**调用 `loss.backward()` 计算梯度**

```python
loss.backward()
```

PyTorch 会根据前向传播记录的计算图，自动使用链式法则计算：

$$
\frac{\partial L}{\partial W},
\qquad
\frac{\partial L}{\partial b}
$$

然后把梯度保存进各参数的：

```python
parameter.grad
```

这一步只是计算梯度，还没有修改参数。

对应你之前的手写代码：

```python
model.backward(grad)
```

---

**调用优化器更新模型参数**

```python
optimizer.step()
```

优化器读取所有参数的 `.grad`，然后更新参数。

以 SGD 为例，大致是：

$$
W \leftarrow W-lr\frac{\partial L}{\partial W}
$$

$$
b \leftarrow b-lr\frac{\partial L}{\partial b}
$$

它对应你之前手写的：

```python
layer.W -= lr * layer.grad_W
layer.b -= lr * layer.grad_b
```

所以这三句是一套：

```python
optimizer.zero_grad()  # 清空旧梯度
loss.backward()        # 计算新梯度
optimizer.step()       # 用梯度更新参数
```

---

### 19.3 Epoch 指标的累计与返回

```python
total_loss += loss.item() * y.size(0)
```

这句要拆开看。

**`loss.item()`：读取标量损失**

```python
loss.item()
```

把只有一个数的 PyTorch 张量转成普通 Python 数字。

例如：

```python
loss
# tensor(0.7234)
```

那么：

```python
loss.item()
# 0.7234
```

**`y.size(0)`：读取当前 Batch 样本数**

当前 batch 的样本数量。

如果：

```
y.shape = (32,)
```

那么：

```python
y.size(0)
```

就是：

```
32
```

**按 Batch 样本数加权损失的原因**

`CrossEntropyLoss` 默认给出的是当前 batch 的平均损失。

假设这一批有 32 个样本：

```
平均损失 = 0.5
```

则这一批所有样本的损失总和为：

```
0.5 × 32 = 16
```

所以先累加损失总和：

```python
total_loss += loss.item() * y.size(0)
```

最终再除以总样本数：

```python
total_loss / total
```

得到整个 epoch 的平均损失。

---

**不能直接累加 Batch 平均损失的原因**

因为最后一个 batch 可能比较小。

例如一共 100 个样本：

```
第 1 批：32
第 2 批：32
第 3 批：32
第 4 批：4
```

如果直接对四个 batch 的平均损失再平均，那么最后 4 个样本会和前面每组 32 个样本占相同权重。

所以要按照每个 batch 的样本数量加权。

---

**统计分类正确的样本数**

```python
total_correct += (logits.argmax(dim=1) == y).sum().item()
```

逐步拆开。

**`logits.argmax(dim=1)`：取得预测类别**

对每一个样本，在所有类别中找评分最大的位置。

例如：

```python
logits = torch.tensor([
    [0.2, 1.5, 0.3],
    [2.0, 0.1, 0.4]
])
```

那么：

```python
logits.argmax(dim=1)
```

得到：

```
tensor([1, 0])
```

表示：

- 第一个样本预测为第 1 类；
- 第二个样本预测为第 0 类。

这里不需要先 Softmax，因为 Softmax 不改变最大值的位置。

**`== y`：比较预测与真实标签**

把预测结果和真实标签逐个比较：

```
预测：[1, 0, 2]
真实：[1, 2, 2]
```

得到：

```
[True, False, True]
```

**`.sum()`：累计正确样本数**

布尔值求和时：

```
True  = 1
False = 0
```

所以结果是：

```
2
```

说明这一批有两个样本预测正确。

最后 `.item()` 把张量变成普通整数。

---

**累计已处理的样本总数**

```python
total += y.size(0)
```

每处理一批，就把这一批的样本数加进去。

最后 `total` 是这一整个 epoch 处理的样本总数。

---

**返回 Epoch 损失与准确率**

```python
return {
    "loss": total_loss / total,
    "accuracy": total_correct / total
}
```

返回的是一个字典，例如：

```python
{
    "loss": 0.427,
    "accuracy": 0.863
}
```

意思是：

```
平均损失：0.427
准确率：86.3%
```

---

**单个 Batch 的训练顺序汇总**

循环内部最核心的流程是：

```python
optimizer.zero_grad()

logits = model(x)
loss = criterion(logits, y)

loss.backward()
optimizer.step()
```

也就是：

```
清梯度
→ 前向传播
→ 计算损失
→ 反向求梯度
→ 更新参数
```

每一个 batch 都会更新一次参数。

所以一个 epoch 内，参数通常会更新很多次，而不是只在整个 epoch 结束后更新一次。

---

### 19.4 Checkpoint 的保存、恢复与训练衔接

下面这段：

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "best_val_loss": best_val_loss,
}, "checkpoint.pt")
```

是在保存训练存档。

可以类比游戏存档：

> 不仅保存角色当前能力，还保存游戏进度以及其他状态。

---

**`model.state_dict()`：模型参数状态**

```python
"model": model.state_dict()
```

保存模型的参数和状态，例如：

```
各层 weight
各层 bias
BatchNorm 的统计信息
```

这部分决定模型当前学到了什么。

---

**`optimizer.state_dict()`：优化器状态**

```python
"optimizer": optimizer.state_dict()
```

保存优化器状态。

例如 Adam 还保存：

```
一阶动量
二阶动量
当前训练步数
```

带 momentum 的 SGD 也有动量状态。

如果想无缝恢复训练，仅保存模型参数往往不够，还应保存优化器状态。

---

**`epoch`：已完成的训练轮次**

```python
"epoch": epoch
```

记录训练到了第几轮。

例如：

```
epoch = 15
```

恢复训练时就知道下一次从第 16 轮开始。

---

**`best_val_loss`：最佳验证损失**

```python
"best_val_loss": best_val_loss
```

记录历史上最好的验证集损失。

这样恢复训练后，还能继续判断新模型是否打破之前的最好成绩。

---

**`checkpoint.pt`：检查点文件路径**

```python
"checkpoint.pt"
```

是保存的文件名。

最终会生成一个 PyTorch 存档文件：

```
checkpoint.pt
```

---

**恢复 Checkpoint 的标准流程**

```python
checkpoint = torch.load("checkpoint.pt")

model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])

start_epoch = checkpoint["epoch"] + 1
best_val_loss = checkpoint["best_val_loss"]
```

这就恢复了：

- 模型参数；
- 优化器状态；
- 训练轮数；
- 最佳验证损失。

---

**训练循环与 Checkpoint 的配合**

训练时通常外面还有一个 epoch 循环：

```python
for epoch in range(num_epochs):
    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device
    )

    print(metrics)

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }, "checkpoint.pt")
```

整体结构就是：

```
第 1 个 epoch
    处理所有 batch
    保存进度

第 2 个 epoch
    处理所有 batch
    保存进度

第 3 个 epoch
    ...
```

一句话概括：

> `train_one_epoch()` 负责用整个训练集训练一轮；checkpoint 负责把训练到当前时刻的模型、优化器和进度保存下来。

## 20. 模型检查点的保存与恢复

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "best_val_loss": best_val_loss,
}, "checkpoint.pt")
```

恢复训练时同时恢复优化器状态。只做推理可只保存模型权重和构建模型所需配置。不要随意加载不可信来源的 pickle 型 checkpoint；优先使用安全权重格式并核验来源。

## 21. 验收标准与知识复盘

- 能不看答案写出四段训练循环。
- 能解释为什么清梯度、为什么评估要 `eval()` 和 `no_grad()`。
- 能从张量形状定位矩阵乘法错误。
- NumPy 反向传播通过数值梯度检查。
- PyTorch 模型保存加载后，同一输入的 logits 在容差内一致。

如果其中任一项无法独立完成，先回到对应讲解和文内例子复盘；不要直接跳到 Project。

## 22. 课后综合项目

完整参考实现见：[模块 2 标准答案](02_NumPy神经网络与PyTorch_标准答案.md)。建议先独立完成，再对照实现定位差异。

只想核对“完整程序从导入到训练结束如何组织”时，可对照：[NumPy 与 PyTorch MLP 完整模型代码](02_NumPy神经网络与PyTorch_完整模型代码.md)。

### 22.1 NumPy MLP 项目

任务：在二维合成分类数据或小型手写数字子集上，实现 `Linear/ReLU/SoftmaxCrossEntropy/SGD`。

必须提交：

- 每层 forward/backward 的形状表。
- 梯度检查脚本。
- 训练/验证损失曲线和决策边界（二维数据）。
- 至少一次消融：去掉 ReLU 或改变隐藏宽度。
- 失败记录：至少解释一个训练不收敛的配置。

### 22.2 PyTorch MNIST/FashionMNIST 项目

任务：用 Dataset、DataLoader、Module、Optimizer 完成训练、验证、推理和保存加载。环境准备请按 [PyTorch 使用指南](../PyTorch使用指南.md) 完成。

建议实验：

1. SGD 与 Adam 的收敛曲线。
2. 隐藏宽度 64/256/1024 的参数量、速度和精度。
3. 随机展示 20 个错误分类，分析是否有共同模式。
4. 固定随机种子后重复运行；说明“可复现”仍可能受硬件算子影响。
