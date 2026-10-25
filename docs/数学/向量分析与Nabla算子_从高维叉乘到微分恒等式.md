# 向量分析与 Nabla 算子：从高维叉乘到微分恒等式

> **学习目标**：不把叉乘、梯度、散度、旋度、拉普拉斯算子当作互不相关的公式，而是从**线性代数、微分算子和几何结构**出发，理解它们的定义、适用范围和运算规律。本文默认已掌握偏导数、链式法则、矩阵乘法和行列式的基本计算，不再复述初等求导。

## 0. 先看整套结构

- **代数层**：点乘把两个向量映射为标量；三维叉乘把两个向量映射为垂直向量；高维更自然的推广是**外积（楔积）**。
- **微分层**：把各坐标方向的偏导运算符组织成 $\nabla$；梯度、散度、旋度是它作用于不同对象后得到的运算。
- **矩阵层**：梯度是标量函数导数的列向量；向量场的导数是 **Jacobian 矩阵**；散度是其迹；旋度对应其反对称部分（在三维可再识别为一个向量）。
- **积分层**：梯度、散度和旋度分别通过梯度定理、高斯定理和斯托克斯定理连接局部微分与整体积分。

---

## 1. 任意维点乘与三维叉乘

### 1.1 点乘：任意维度都成立

令 $\mathbf a,\mathbf b\in\mathbb R^n$，把它们看成列向量，则

$$
\boxed{\mathbf a\cdot\mathbf b=\mathbf a^{\mathsf T}\mathbf b
=\sum_{i=1}^n a_i b_i.}
$$

这是 $(1\times n)$ 矩阵与 $(n\times1)$ 矩阵的乘积，输出一个标量。它的几何意义是

$$
\mathbf a\cdot\mathbf b=\|\mathbf a\|\,\|\mathbf b\|\cos\theta.
$$

注意：点乘并不依赖三维空间，所以从二维推广到 $n$ 维不需要创造新运算。

### 1.2 三维叉乘为什么是那三个分量？

在给定长度、正交关系和右手取向的三维欧氏空间中，叉乘 $\mathbf a\times\mathbf b$ 满足：

1. 与 $\mathbf a,\mathbf b$ 都垂直；
2. 长度为 $\|\mathbf a\|\,\|\mathbf b\|\sin\theta$；
3. 方向由右手定则确定；
4. 对两个输入分别线性。

在标准正交基 $\mathbf e_1,\mathbf e_2,\mathbf e_3$ 下，规定

$$
\mathbf e_1\times\mathbf e_2=\mathbf e_3,\qquad
\mathbf e_2\times\mathbf e_3=\mathbf e_1,\qquad
\mathbf e_3\times\mathbf e_1=\mathbf e_2,
$$

且 $\mathbf e_i\times\mathbf e_i=0$，交换两因子会改变符号。将

$$
\mathbf a=\sum_i a_i\mathbf e_i,\qquad
\mathbf b=\sum_j b_j\mathbf e_j
$$

代入，利用双线性展开并合并同类项：

$$
\boxed{
\mathbf a\times\mathbf b=
\begin{pmatrix}
 a_2b_3-a_3b_2\\
 a_3b_1-a_1b_3\\
 a_1b_2-a_2b_1
\end{pmatrix}.}
$$

因此，分量公式是从**基向量之间的乘法关系 + 双线性**推出来的，不是从行列式中凭空规定的。常见的行列式记法

$$
\mathbf a\times\mathbf b=
\begin{vmatrix}
\mathbf e_1&\mathbf e_2&\mathbf e_3\\
a_1&a_2&a_3\\
b_1&b_2&b_3
\end{vmatrix}
$$

只是上述展开规律的简洁记忆形式。

### 1.3 叉乘怎样写成真正的矩阵乘法？

固定 $\mathbf a$ 后，映射 $\mathbf b\mapsto\mathbf a\times\mathbf b$ 是线性的，因此必然能由一个矩阵表示：

$$
[\mathbf a]_\times=
\begin{pmatrix}
0&-a_3&a_2\\
a_3&0&-a_1\\
-a_2&a_1&0
\end{pmatrix},\qquad
\boxed{\mathbf a\times\mathbf b=[\mathbf a]_\times\mathbf b.}
$$

例如 $\mathbf a=(1,2,3)^{\mathsf T}$、$\mathbf b=(4,5,6)^{\mathsf T}$：

$$
\begin{pmatrix}
0&-3&2\\3&0&-1\\-2&1&0
\end{pmatrix}
\begin{pmatrix}4\\5\\6\end{pmatrix}
=
\begin{pmatrix}-3\\6\\-3\end{pmatrix}.
$$

这里 $[\mathbf a]_\times^{\mathsf T}=-[\mathbf a]_\times$，它是**反对称矩阵**。这个性质并非巧合：它将在高维推广及旋度的矩阵表示中再次出现。

---

## 2. 高维空间为什么不能直接沿用三维叉乘？

### 2.1 先看几何上的特殊性

设 $\mathbf a,\mathbf b\in\mathbb R^n$ 线性无关。如果希望输出向量 $\mathbf c$ 同时满足

$$
\mathbf c\cdot\mathbf a=0,\qquad
\mathbf c\cdot\mathbf b=0,
$$

那么 $\mathbf c$ 属于两个向量张成平面的正交补。由秩—零度定理，正交补维数为

$$
\boxed{\dim\big(\operatorname{span}\{\mathbf a,\mathbf b\}^{\perp}\big)=n-2.}
$$

- **三维**：$n-2=1$。垂直方向是一条直线；再用长度和取向即可确定一个向量。
- **四维**：$n-2=2$。垂直方向构成一个二维平面；仅凭「同时垂直、长度指定」并不能选出唯一方向。
- **五维及以上**：正交补维数更高，同样没有自然的唯一法向方向。

这解释了**为什么三维叉乘的几何定义不能直接照搬**。但请注意：**“垂直方向不唯一”并不能严格证明其他维度绝不可能构造叉乘**；如果额外赋予空间特殊的代数结构，还可能选出方向。

一个重要例外是**七维叉乘**。若要求二元运算满足双线性、两输入正交，以及

$$
\|\mathbf a\times\mathbf b\|^2
=\|\mathbf a\|^2\|\mathbf b\|^2-(\mathbf a\cdot\mathbf b)^2,
$$

那么非平凡的欧氏向量叉乘可存在于三维和七维；七维构造与**八元数（octonions）**有关，但不再是只由通常的度量和取向便自然确定的三维叉乘。因此，通常所说“叉乘只在三维成立”，准确含义是**标准的、自然的二元向量叉乘是三维特有的形式**，而不是数学上绝无例外。

### 2.2 第一种推广：外积（楔积），不强行让结果是向量

三维叉乘的模长实际上反映的是两个向量张成的**有向面积**。在 $n$ 维中，与其硬找一个垂直向量，不如直接记录各坐标平面中的有向面积：

$$
\boxed{\mathbf a\wedge\mathbf b
=\sum_{1\le i<j\le n}(a_i b_j-a_j b_i)\,\mathbf e_i\wedge\mathbf e_j.}
$$

这称为**外积（楔积）**，结果是**二向量（bivector）**，不是一般意义下的 $n$ 维向量。

对四维向量 $\mathbf a=(a_1,a_2,a_3,a_4)$、$\mathbf b=(b_1,b_2,b_3,b_4)$，独立分量对应坐标对

$$
(1,2),\ (1,3),\ (1,4),\ (2,3),\ (2,4),\ (3,4),
$$

因此总共有

$$
\boxed{\binom n2=\frac{n(n-1)}2}
$$

个独立分量；四维时为 $6$ 个。

还可以把这些信息整理成反对称矩阵

$$
\boxed{W=\mathbf a\mathbf b^{\mathsf T}-\mathbf b\mathbf a^{\mathsf T},
\qquad W_{ij}=a_i b_j-a_j b_i.}
$$

因为 $W_{ji}=-W_{ij}$ 且 $W_{ii}=0$，一个 $n\times n$ 矩阵中恰好只有 $n(n-1)/2$ 个独立数。

**外积如何保持面积意义？** 利用恒等式

$$
\begin{aligned}
\sum_{i<j}(a_i b_j-a_j b_i)^2
&=\frac12\sum_{i,j}(a_i b_j-a_j b_i)^2\\
&=\|\mathbf a\|^2\|\mathbf b\|^2-(\mathbf a\cdot\mathbf b)^2\\
&=\|\mathbf a\|^2\|\mathbf b\|^2\sin^2\theta.
\end{aligned}
$$

因此给二向量采用自然的长度定义后，

$$
\boxed{\|\mathbf a\wedge\mathbf b\|=\|\mathbf a\|\|\mathbf b\|\sin\theta.}
$$

可见三维叉乘中“面积”的几何含义被完整保留下来了，变化的只是**输出对象的类型**。

### 2.3 为什么三维的外积又能变回向量？

在三维中，二向量空间恰好也是三维：

$$
\binom32=3.
$$

给定欧氏内积和空间取向，可以用 **Hodge 星算子** $\star$ 将一个二向量转换为一个向量（严格地说，需要配合由内积提供的向量与协向量对应）：

$$
\boxed{\mathbf a\times\mathbf b\ \longleftrightarrow\ \star(\mathbf a\wedge\mathbf b).}
$$

在标准右手基下，这个对应将

$$
(W_{23},\ W_{31},\ W_{12})
$$

识别为通常叉乘的三个分量。四维的二向量有六个独立分量，不可能通过同样的自然对应变成一个四维向量。

用更一般的语言说，$\star$ 会把 $k$ 阶的有向几何对象变成 $n-k$ 阶对象：**三维时 $2\mapsto1$，四维时 $2\mapsto2$**。这才是三维叉乘特别方便的深层原因。

### 2.4 第二种推广：让 $n-1$ 个向量产生一个垂直向量

如果确实希望输出一个 $n$ 维向量，那么可以改变输入个数。给定 $n-1$ 个向量 $\mathbf v_1,\ldots,\mathbf v_{n-1}$，定义广义叉乘 $\mathbf c$ 满足：对任意 $\mathbf u$，

$$
\boxed{\mathbf u\cdot\mathbf c
=\det\big[\mathbf v_1\ \mathbf v_2\ \cdots\ \mathbf v_{n-1}\ \mathbf u\big].}
$$

行列式关于 $\mathbf u$ 是线性的，因此存在唯一 $\mathbf c$ 实现这个等式；取 $\mathbf u=\mathbf v_j$ 时，行列式有两列相同，得到 $\mathbf v_j\cdot\mathbf c=0$。

- 在三维，输入 $2$ 个向量，就恢复通常叉乘。
- 在四维，输入 $3$ 个向量；例如标准取向下，$\mathbf e_1,\mathbf e_2,\mathbf e_3$ 的广义叉乘是 $\mathbf e_4$。

它与双向量外积推广解决的是**不同的问题**，不能混为一谈。

---

## 3. $\nabla$ 为什么能参与点乘、叉乘？

### 3.1 $\nabla$ 是微分算子的“向量式排列”

在 $n$ 维直角坐标系中记

$$
D_i:=\frac{\partial}{\partial x_i},\qquad
\boxed{\nabla=(D_1,\ldots,D_n).}
$$

$D_i$ 不是普通数字，而是作用在函数上的**线性算子**。因此，$\nabla$ 可以采用向量和矩阵记法，但不能毫无条件地当成常数向量进行代数移项。

尤其要区分：

$$
\nabla f\quad\text{（函数 $f$ 的梯度）},
\qquad f\nabla\quad\text{（先乘 $f$，再执行的微分算子）}.
$$

它们不是相同的对象。要弄清楚原因，引入乘法算子 $M_f$：

$$
M_f(g)=fg.
$$

对任意足够光滑的测试函数 $g$，按乘积法则有

$$
\begin{aligned}
(D_iM_f)(g)
&=D_i(fg)\\
&=(D_if)g+fD_ig\\
&=(M_{D_if}+M_fD_i)(g).
\end{aligned}
$$

因为对任意 $g$ 都成立，故有**算子恒等式**

$$
\boxed{D_iM_f=M_fD_i+M_{D_if}.}
$$

这说明：**偏导算子与乘法算子通常不交换**。实际上，后面所有含 $\nabla$ 的乘积法则都源于这个关系，而不是因为 $\nabla$ 被“允许像普通向量一样求导”。

### 3.2 梯度、Jacobian 与散度：矩阵观点

对标量函数 $f:\mathbb R^n\to\mathbb R$，梯度是

$$
\boxed{\nabla f=(D_1f,\ldots,D_nf)^{\mathsf T}.}
$$

对向量场 $\mathbf F=(F_1,\ldots,F_n)^{\mathsf T}$，定义 **Jacobian 矩阵**

$$
\boxed{J_{\mathbf F}=(D_jF_i)_{i,j=1}^n
=\begin{pmatrix}
D_1F_1&\cdots&D_nF_1\\
\vdots&\ddots&\vdots\\
D_1F_n&\cdots&D_nF_n
\end{pmatrix}.}
$$

它把小位移 $d\mathbf x$ 近似映射成向量场的增量：

$$
d\mathbf F\approx J_{\mathbf F}\,d\mathbf x.
$$

散度恰好是这个矩阵的迹（对角元素之和）：

$$
\boxed{\nabla\cdot\mathbf F
=\sum_{i=1}^n D_iF_i
=\operatorname{tr}(J_{\mathbf F}).}
$$

因此 $\nabla\cdot\mathbf F$ 的“点乘”形式是对**相应分量求偏导再求和**的简记，并不表示微分算子与数值向量之间存在普通的数值内积。

还有一种容易混淆的写法：

$$
\boxed{\mathbf F\cdot\nabla=\sum_{i=1}^nF_iD_i.}
$$

它是一个新的微分算子，作用于 $g$ 时得到方向导数 $\mathbf F\cdot\nabla g$；而 $\nabla\cdot\mathbf F$ 是一个标量函数。例如 $\mathbf F=(x,0,0)$ 时，

$$
\nabla\cdot\mathbf F=1,\qquad
(\mathbf F\cdot\nabla)g=x\frac{\partial g}{\partial x}.
$$

二者的输出类型完全不同，不能交换位置随意改写。

### 3.3 旋度为何能写成行列式？

在三维中，定义

$$
\boxed{\nabla\times\mathbf F=
\begin{pmatrix}
D_2F_3-D_3F_2\\
D_3F_1-D_1F_3\\
D_1F_2-D_2F_1
\end{pmatrix}.}
$$

我们可以形式上写作

$$
\nabla\times\mathbf F=
\begin{vmatrix}
\mathbf e_1&\mathbf e_2&\mathbf e_3\\
D_1&D_2&D_3\\
F_1&F_2&F_3
\end{vmatrix},
$$

但应将它视为**分量定义的记忆式**：$D_i$ 必须作用到右侧的函数，不能把它当作普通数字行列式并忽略算子次序。

再看与高维的联系。以 $J_{\mathbf F}$ 为前文约定的 Jacobian，构造反对称矩阵

$$
\boxed{B=J_{\mathbf F}^{\mathsf T}-J_{\mathbf F},\qquad
B_{ij}=D_iF_j-D_jF_i.}
$$

在三维，$B$ 的三个独立分量恰好是

$$
(\nabla\times\mathbf F)_1=B_{23},\qquad
(\nabla\times\mathbf F)_2=B_{31},\qquad
(\nabla\times\mathbf F)_3=B_{12}.
$$

**这就是旋度与高维外积的对应关系**：三维的反对称微分信息能被编码成三分量向量；到 $n$ 维时，$B$ 仍然有 $n(n-1)/2$ 个独立分量，通常不能再用单个 $n$ 维向量完整表示。

例如在四维中，可以保留六个分量 $D_iF_j-D_jF_i$（$i<j$），这就是旋度自然的高维版本之一。在微分形式语言下，它是向量场对应的 $1$-形式的**外微分** $d(F^\flat)$。

---

## 4. $\nabla$ 的运算律：从定义推导，不靠背诵

以下假定函数具有所需阶数的连续偏导数。

### 4.1 梯度的乘积法则

逐分量利用 $D_i(fg)=(D_if)g+f(D_ig)$：

$$
\begin{aligned}
\nabla(fg)
&=(D_1(fg),\ldots,D_n(fg))^{\mathsf T}\\
&=g(D_1f,\ldots,D_nf)^{\mathsf T}
 +f(D_1g,\ldots,D_ng)^{\mathsf T}.
\end{aligned}
$$

所以

$$
\boxed{\nabla(fg)=g\nabla f+f\nabla g.}
$$

### 4.2 散度的乘积法则

令 $f$ 为标量场，$\mathbf F$ 为向量场：

$$
\begin{aligned}
\nabla\cdot(f\mathbf F)
&=\sum_iD_i(fF_i)\\
&=\sum_i(D_if)F_i+f\sum_iD_iF_i\\
&=\nabla f\cdot\mathbf F+f(\nabla\cdot\mathbf F).
\end{aligned}
$$

于是

$$
\boxed{\nabla\cdot(f\mathbf F)
=\nabla f\cdot\mathbf F+f\nabla\cdot\mathbf F.}
$$

### 4.3 旋度的乘积法则

仍在三维，仅展开第一分量就能看出结构：

$$
\begin{aligned}
[\nabla\times(f\mathbf F)]_1
&=D_2(fF_3)-D_3(fF_2)\\
&=(D_2f)F_3-(D_3f)F_2
 +f(D_2F_3-D_3F_2)\\
&=[\nabla f\times\mathbf F]_1
 +[f(\nabla\times\mathbf F)]_1.
\end{aligned}
$$

另两个分量循环置换指标，得到

$$
\boxed{\nabla\times(f\mathbf F)
=\nabla f\times\mathbf F+f(\nabla\times\mathbf F).}
$$

可见规则来自偏导乘积法则，而非“把 $\nabla$ 当成数值向量直接分配”。

### 4.4 一个验证算例

取 $f=x$，$\mathbf F=(-y,x,0)$。此时

$$
\nabla f=(1,0,0),\quad
\nabla\cdot\mathbf F=0,\quad
\nabla\times\mathbf F=(0,0,2).
$$

于是

$$
\nabla\cdot(f\mathbf F)=\nabla f\cdot\mathbf F=-y,
$$

$$
\nabla\times(f\mathbf F)
=(1,0,0)\times(-y,x,0)+x(0,0,2)=(0,0,3x).
$$

直接对 $f\mathbf F=(-xy,x^2,0)$ 求散度和旋度，得到相同结果。

---

## 5. 二阶算子与核心恒等式

### 5.1 拉普拉斯算子为什么是“散度的梯度”？

标量场 $f$ 的梯度是向量场，因此还可以对它取散度：

$$
\begin{aligned}
\nabla\cdot(\nabla f)
&=\sum_iD_i(D_if)\\
&=\sum_i\frac{\partial^2f}{\partial x_i^2}.
\end{aligned}
$$

定义

$$
\boxed{\Delta f=\nabla^2f
=\nabla\cdot\nabla f
=\sum_{i=1}^n\frac{\partial^2f}{\partial x_i^2}.}
$$

又因为 Hessian 矩阵 $H_f=(D_jD_if)_{i,j}$，故

$$
\boxed{\Delta f=\operatorname{tr}(H_f).}
$$

例：在 $n$ 维中，$f(\mathbf x)=\|\mathbf x\|^2=\sum_i x_i^2$，那么

$$
\boxed{\Delta\|\mathbf x\|^2=2n.}
$$

这清楚显示拉普拉斯算子自然存在于任意维度。

### 5.2 为什么梯度的旋度为零？

三维中

$$
\nabla\times(\nabla f)
=\begin{pmatrix}
D_2D_3f-D_3D_2f\\
D_3D_1f-D_1D_3f\\
D_1D_2f-D_2D_1f
\end{pmatrix}.
$$

在 $f\in C^2$ 的条件下，混合二阶偏导数可交换，因此

$$
\boxed{\nabla\times(\nabla f)=\mathbf 0.}
$$

这里“等于零”的根源是**Hessian 对称**；高维版本是 $d(df)=0$。

### 5.3 为什么旋度的散度为零？

将旋度分量代入：

$$
\begin{aligned}
\nabla\cdot(\nabla\times\mathbf F)
={}&D_1(D_2F_3-D_3F_2)\\
&+D_2(D_3F_1-D_1F_3)\\
&+D_3(D_1F_2-D_2F_1).
\end{aligned}
$$

六项两两抵消（要求相应二阶混合偏导数可交换），所以

$$
\boxed{\nabla\cdot(\nabla\times\mathbf F)=0.}
$$

### 5.4 向量拉普拉斯与“旋度的旋度”

直角坐标系中，对向量场逐分量定义

$$
\Delta\mathbf F=(\Delta F_1,\Delta F_2,\Delta F_3).
$$

利用 Levi-Civita 符号 $\varepsilon_{ijk}$，三维旋度可写作

$$
(\nabla\times\mathbf F)_i=\varepsilon_{ijk}D_jF_k
\qquad(\text{重复指标求和}).
$$

再取一次旋度，并使用

$$
\varepsilon_{ijk}\varepsilon_{k\ell m}
=\delta_{i\ell}\delta_{jm}-\delta_{im}\delta_{j\ell},
$$

得到

$$
\begin{aligned}
[\nabla\times(\nabla\times\mathbf F)]_i
&=\varepsilon_{ijk}D_j\big(\varepsilon_{k\ell m}D_\ell F_m\big)\\
&=D_i(D_jF_j)-D_jD_jF_i\\
&=[\nabla(\nabla\cdot\mathbf F)]_i-(\Delta\mathbf F)_i.
\end{aligned}
$$

因此

$$
\boxed{\Delta\mathbf F
=\nabla(\nabla\cdot\mathbf F)
-\nabla\times(\nabla\times\mathbf F).}
$$

这条公式不是把三个 $\nabla$ 当成普通向量随便应用“三重积公式”得来的，而是对分量进行求导、合并后成立的算子恒等式。

---

## 6. 微分算子与积分定理为什么成对出现？

下列公式在适当的光滑性、区域正则性和取向约定下成立。

### 6.1 梯度 ↔ 沿路径的变化量

把曲线参数化为 $\mathbf r(t)$，$t\in[a,b]$，并令 $\mathbf r(a)=A,\mathbf r(b)=B$。由链式法则：

$$
\frac{d}{dt}f(\mathbf r(t))
=\nabla f(\mathbf r(t))\cdot\mathbf r'(t).
$$

沿曲线积分，得到

$$
\boxed{\int_C\nabla f\cdot d\mathbf r=f(B)-f(A).}
$$

所以梯度场沿路径积分只与端点有关（前提是 $f$ 在路径上定义且足够光滑）。

### 6.2 散度 ↔ 穿过封闭曲面的净通量

考虑一个小长方体 $\Delta V=\Delta x\Delta y\Delta z$。其两侧 $x$ 面上的净通量近似为

$$
\big[F_1(x+\Delta x,y,z)-F_1(x,y,z)\big]\Delta y\Delta z
\approx (D_1F_1)\Delta V.
$$

对 $y,z$ 方向重复，得到小体元总净通量近似等于

$$
(\nabla\cdot\mathbf F)\Delta V.
$$

将区域分成许多小体元求和时，**内部公共面上的通量方向相反并相互抵消**，最终只留下外边界；在适当条件下取极限，就得到高斯散度定理：

$$
\boxed{\iiint_V\nabla\cdot\mathbf F\,dV
=\oiint_{\partial V}\mathbf F\cdot\mathbf n\,dS.}
$$

这里 $\mathbf n$ 是外法向单位向量。

### 6.3 旋度 ↔ 沿闭合边界的环流

在 $xy$ 平面取一小矩形（法向为 $+\mathbf e_3$），沿逆时针方向计算向量场 $\mathbf F=(P,Q,R)$ 的环流。上下两边贡献 $P$ 的差，左右两边贡献 $Q$ 的差，合并得

$$
\oint_{\partial S}\mathbf F\cdot d\mathbf r
\approx (D_1Q-D_2P)\Delta x\Delta y
=(\nabla\times\mathbf F)_3\Delta S.
$$

把曲面细分后，**内部公共边上的环流方向相反并相互抵消**，只留下总边界；取极限可得斯托克斯定理：

$$
\boxed{\oint_{\partial S}\mathbf F\cdot d\mathbf r
=\iint_S(\nabla\times\mathbf F)\cdot\mathbf n\,dS.}
$$

边界方向和法向必须满足右手取向约定。

由此可以看到三个关系的统一结构：**局部微分量积分起来，等于边界上累积的整体变化**。

---

## 7. 坐标系改变时，为什么公式看起来不一样？

前面紧凑的写法依赖**笛卡尔坐标中基向量固定不变**。在柱坐标、球坐标中，基向量随位置变化，而且坐标增量和实际长度之间存在尺度因子，因此不能简单地用 $\partial_r,\partial_\theta$ 等原样拼出所有公式。

一个最基本的矩阵版链式法则是：若 $\mathbf y=T(\mathbf x)$，$h(\mathbf x)=f(T(\mathbf x))$，则

$$
\boxed{\nabla_{\mathbf x}h
=J_T(\mathbf x)^{\mathsf T}\nabla_{\mathbf y}f\big|_{\mathbf y=T(\mathbf x)}.}
$$

这是逐分量链式法则的矩阵形式，说明**梯度如何随着坐标变换而改变表达**。对于正交曲线坐标，还需处理基向量与尺度因子，不能只替换坐标字母。

例如三维柱坐标 $(r,\varphi,z)$ 中，标量拉普拉斯算子为

$$
\boxed{\Delta f=
\frac1r\frac{\partial}{\partial r}
\left(r\frac{\partial f}{\partial r}\right)
+\frac1{r^2}\frac{\partial^2 f}{\partial\varphi^2}
+\frac{\partial^2 f}{\partial z^2}.}
$$

它并不等于简单的 $f_{rr}+f_{\varphi\varphi}+f_{zz}$；额外因子反映的正是坐标几何。

---

## 8. 最后整理：一张速查表

| 对象/运算 | 输入 → 输出 | 一般维度下的核心表达 |
|---|---|---|
| 点乘 | 两个 $n$ 维向量 → 标量 | $\mathbf a^{\mathsf T}\mathbf b$ |
| 三维叉乘 | 两个 $3$ 维向量 → $3$ 维向量 | $\mathbf a\times\mathbf b=[\mathbf a]_\times\mathbf b$ |
| 楔积 | 两个 $n$ 维向量 → 二向量 | $\sum_{i<j}(a_ib_j-a_jb_i)\mathbf e_i\wedge\mathbf e_j$ |
| 广义叉乘 | $n-1$ 个 $n$ 维向量 → 向量 | $\mathbf u\cdot\mathbf c=\det[\mathbf v_1\cdots\mathbf v_{n-1}\ \mathbf u]$ |
| 梯度 | 标量场 → 向量场 | $(D_1f,\ldots,D_nf)^{\mathsf T}$ |
| Jacobian | 向量场 → 矩阵场 | $(D_jF_i)_{ij}$ |
| 散度 | 向量场 → 标量场 | $\operatorname{tr}(J_{\mathbf F})$ |
| 高维旋转微分信息 | 向量场 → 反对称矩阵（或 $2$-形式） | $B_{ij}=D_iF_j-D_jF_i$ |
| 标量拉普拉斯 | 标量场 → 标量场 | $\Delta f=\operatorname{tr}(H_f)$ |

### 最容易犯的三个错误

1. **以为四维完全不能描述两个向量张成的有向面积。** 实际上可用楔积，四维会得到六个独立分量；不能自然得到的只是三维那种单个法向量表示。
2. **把 $\nabla$ 视为普通常数向量。** 其分量是作用于函数的微分算子；算子与乘法算子不交换，因而必须使用乘积法则。
3. **把 $\mathbf F\cdot\nabla$ 与 $\nabla\cdot\mathbf F$ 混淆。** 前者是微分算子，后者是函数；写法相似但作用方式不同。

### 建议的自测

1. 对 $\mathbf a=(1,0,1,2)$、$\mathbf b=(0,1,2,1)$，构造 $W=\mathbf a\mathbf b^{\mathsf T}-\mathbf b\mathbf a^{\mathsf T}$，验证它有六个独立反对称分量，且 $\sum_{i<j}W_{ij}^2=\|\mathbf a\|^2\|\mathbf b\|^2-(\mathbf a\cdot\mathbf b)^2$。
2. 从 $D_iM_f=M_fD_i+M_{D_if}$ 出发，不查表推导 $\nabla\cdot(f\mathbf F)$ 和 $\nabla\times(f\mathbf F)$。
3. 令 $\mathbf F=(-y,x,0)$，分别算 $J_{\mathbf F}$、$\operatorname{tr}(J_{\mathbf F})$ 和 $J_{\mathbf F}^{\mathsf T}-J_{\mathbf F}$，说明散度与旋度捕捉的是 Jacobian 的什么部分。
4. 利用二阶混合偏导可交换，推导 $\nabla\times(\nabla f)=0$。思考为什么若混合偏导不可交换就不能直接使用此结论。

> **一句话总结**：点乘来自内积；三维叉乘是有向面积在三维中可以自然编码为法向量的特殊结果；高维用楔积与反对称对象延拓；而 $\nabla$ 的各种运算律都可以还原为普通偏导算子的作用规律、矩阵结构和混合偏导的性质。
