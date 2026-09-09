# PyTorch 从零使用指南

PyTorch 可以先理解成两部分：

1. **带 GPU 加速的 NumPy**：用 `Tensor` 做矩阵运算。
2. **自动求导的神经网络框架**：自动计算梯度、更新模型参数。

官方文档目前把入门流程概括为：Tensor、Dataset/DataLoader、构建模型、自动求导、优化模型、保存与加载。

---

## 一、安装 PyTorch

推荐在独立的 Conda 环境中学习 PyTorch。这样每个项目都有自己的 Python 和依赖包，不会把包混装到 `base` 环境，也不会影响其他项目。

### 0. 使用 Conda 创建学习环境（推荐）

先在 **Anaconda Prompt**、PowerShell 或 VS Code 终端中检查 Conda 是否可用：

```powershell
conda --version
```

如果提示“`conda` 不是内部或外部命令”，优先打开 **Anaconda Prompt** 再执行后续命令。若想在普通 PowerShell 中使用，可在 Anaconda Prompt 中一次性运行 `conda init powershell`，关闭并重新打开 PowerShell 后再试；这是为 PowerShell 配置 Conda 初始化脚本。

能显示版本号后，创建并激活一个专门学习 PyTorch 的环境：

```powershell
conda create -n pytorch-study python=3.11 -y
conda activate pytorch-study
python --version
```

这里的 `pytorch-study` 是环境名称；你可以改名，但后续激活时必须使用同一个名字。`python=3.11` 表示在这个隔离环境中安装 Python 3.11；如果课程明确要求 Python 3.10，可以改成 `python=3.10`。

之后每次安装包或运行本项目，先执行：

```powershell
conda activate pytorch-study
```

Conda 负责隔离环境，`pip` 可以在这个 Conda 环境内部安装 PyTorch。下面统一使用 `python -m pip`，而不是直接写 `pip`，这样更容易保证包被安装到当前已激活环境的 Python 中。

如果你暂时没有 Conda，也可以使用下面原有的 `venv` 方案；两者二选一，不要在同一个项目中嵌套使用。

### 备用：使用 venv 创建环境

```powershell
mkdir pytorch_demo
cd pytorch_demo

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

### 1. 安装 CPU 版本

确认终端前缀是 `(pytorch-study)`（或已激活 `.venv`）后执行：

```powershell
python -m pip install torch torchvision torchaudio
```

### 2. 安装 NVIDIA GPU 版本

只有 NVIDIA 显卡才能使用 CUDA。先在已激活的 Conda 环境中检查驱动：

```powershell
nvidia-smi
```

如果该命令能显示显卡名称和驱动版本，再进入 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)。不要随便复制网上旧命令；应根据：

- Windows
- Pip
- Python
- 你的 CUDA 平台

选择对应命令。官方也建议通过安装选择器生成当前适用的命令。

命令通常形如：

```powershell
# 只作格式示意；cuXXX 必须以官网当前给出的值为准
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
```

不要从旧教程里照抄 `cu121`、`cu124` 等具体数字；可用版本会变化。多数学习任务不需要单独安装完整 CUDA Toolkit，PyTorch 的安装包通常已经带有运行所需组件；只有编译自定义 CUDA 扩展时才可能需要 Toolkit。

### 3. Conda、pip 与旧安装命令的关系

有些旧资料会写 `conda install pytorch ... -c pytorch`。是否能用取决于 PyTorch 当前的发布方式；优先以官网安装选择器给出的命令为准。

- 官网给 Conda 命令：在已激活的 `pytorch-study` 环境中运行。
- 官网给 Pip 命令：仍在这个 Conda 环境中运行 `python -m pip ...`，这是正常用法。
- 同一环境不要把 `torch` 的一部分从 Conda 安装、另一部分从 Pip 安装；若出现冲突，删除环境后按一种方案重建通常最快。

安装后检查：

```python
import torch

print("PyTorch 版本：", torch.__version__)
print("CUDA 可用：", torch.cuda.is_available())

if torch.cuda.is_available():
    print("显卡：", torch.cuda.get_device_name(0))
```

可能输出：

```
PyTorch 版本： 2.x.x+cu...
CUDA 可用： True
显卡： NVIDIA GeForce RTX ...
```

这里：

```python
torch.cuda.is_available()
```

返回布尔值：

- `True`：PyTorch 能使用 NVIDIA GPU。
- `False`：目前只能使用 CPU。

若怀疑“明明安装了却导入不到”，在已经执行 `conda activate pytorch-study` 的终端中运行：

```powershell
where python
python -m pip --version
```

两条输出中的路径都应当指向 `pytorch-study` 环境。随后可以再运行下面的最小计算检查：

```python
import torch

x = torch.randn(2, 3)
linear = torch.nn.Linear(3, 4)
y = linear(x)
print(y.shape)  # 应为 torch.Size([2, 4])
```

在 Jupyter 或 VS Code Notebook 中，还要确认右上角选择的内核也是这个 Conda 环境；终端激活环境不会自动替 Notebook 更换内核。

---

# 二、Tensor：PyTorch 最核心的数据结构

Tensor 就是一个多维数字数组。

| 维⁠数 | 示⁠例 | 常⁠见⁠含⁠义 |
| --- | --- | --- |
| 0 维 | `5` | 一⁠个⁠数⁠字 |
| 1 维 | `[1, 2, 3]` | 向⁠量 |
| 2 维 | `[[1, 2], [3, 4]]` | 矩⁠阵 |
| 3 维 | 多⁠张⁠二⁠维⁠表 | 彩⁠色⁠图⁠片⁠或⁠批⁠次 |
| 更⁠高⁠维 | 多⁠层⁠嵌⁠套 | 图⁠片⁠批⁠次、文⁠本⁠批⁠次 |

## 1. 创建 Tensor

```python
import torch

a = torch.tensor([1, 2, 3])
print(a)
```

输出：

```
tensor([1, 2, 3])
```

二维 Tensor：

```python
x = torch.tensor([
    [1, 2],
    [3, 4]
])

print(x)
```

输出：

```
tensor([[1, 2],
        [3, 4]])
```

## 2. 常用创建方法

```python
import torch

a = torch.zeros(2, 3)
b = torch.ones(2, 3)
c = torch.rand(2, 3)
d = torch.randn(2, 3)
e = torch.arange(0, 10, 2)

print(a)
print(b)
print(c)
print(d)
print(e)
```

含义：

```python
torch.zeros(2, 3)
```

创建一个 2 行 3 列、元素全是 0 的 Tensor。

```python
torch.rand(2, 3)
```

创建一个 2 行 3 列的 Tensor，元素来自 `[0, 1)` 均匀分布。

```python
torch.randn(2, 3)
```

元素来自均值为 0、标准差为 1 的正态分布。

```python
torch.arange(0, 10, 2)
```

类似 Python 的：

```python
range(0, 10, 2)
```

得到：

```
tensor([0, 2, 4, 6, 8])
```

---

# 三、Tensor 的形状

```python
x = torch.randn(4, 3)

print(x)
print(x.shape)
print(x.ndim)
print(x.numel())
```

输出可能是：

```
torch.Size([4, 3])
2
12
```

分别表示：

- `x.shape`：4 行 3 列。
- `x.ndim`：二维 Tensor。
- `x.numel()`：总共 12 个元素。

在神经网络中，**形状非常重要**。大量 PyTorch 报错，本质上都是形状不匹配。

---

# 四、Tensor 的数据类型

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([1.0, 2.0, 3.0])

print(a.dtype)
print(b.dtype)
```

通常输出：

```
torch.int64
torch.float32
```

神经网络的大部分计算使用浮点数：

```python
x = torch.tensor([1, 2, 3], dtype=torch.float32)
```

或者转换：

```python
x = x.float()
```

常见类型：

| 类⁠型 | 含⁠义 |
| --- | --- |
| `torch.float32` | 普⁠通⁠浮⁠点⁠数，训⁠练⁠最⁠常⁠用 |
| `torch.float16` | 半⁠精⁠度⁠浮⁠点⁠数 |
| `torch.bfloat16` | 另⁠一⁠种⁠低⁠精⁠度⁠浮⁠点⁠数 |
| `torch.int64` | 整⁠数，分⁠类⁠标⁠签⁠常⁠用 |
| `torch.bool` | 布⁠尔⁠值 |

一个非常常见的规则是：

```python
输入 x：float32
分类标签 y：int64
```

---

# 五、索引与切片

```python
x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print(x[0])
print(x[0, 1])
print(x[:, 0])
print(x[1:, :2])
```

分别得到：

```python
x[0]
```

第一行：

```
tensor([10, 20, 30])
```

```python
x[0, 1]
```

第 0 行、第 1 列，也就是：

```
tensor(20)
```

```python
x[:, 0]
```

所有行的第 0 列：

```
tensor([10, 40, 70])
```

其中：

- `:` 表示这一维全部选择。
- `1:` 表示从索引 1 到最后。
- `:2` 表示从开头到索引 2，但不包括索引 2。

---

# 六、Tensor 运算

## 1. 对应元素运算

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)
print(a - b)
print(a * b)
print(a / b)
```

注意：

```python
a * b
```

是对应元素相乘，不是矩阵乘法。

## 2. 矩阵乘法

```python
a = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

b = torch.tensor([
    [5.0, 6.0],
    [7.0, 8.0]
])

c = a @ b
print(c)
```

也可以写：

```python
c = torch.matmul(a, b)
```

神经网络里的线性层，本质上包含类似：

$$
Y=XW^T+b
$$

的矩阵运算。

## 3. 常用统计运算

```python
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

print(x.sum())
print(x.mean())
print(x.max())
print(x.sum(dim=0))
print(x.sum(dim=1))
```

其中：

```python
x.sum(dim=0)
```

把第 0 维消掉，也就是按列相加：

```
tensor([5., 7., 9.])
```

```python
x.sum(dim=1)
```

把第 1 维消掉，也就是每一行内部相加：

```
tensor([ 6., 15.])
```

可以把 `dim` 理解为：

> 指定“沿着哪一个方向压缩”。

---

# 七、改变 Tensor 形状

## 1. reshape

```python
x = torch.arange(12)

y = x.reshape(3, 4)

print(x)
print(y)
```

输出：

```
tensor([ 0, 1, 2, ..., 11])

tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
```

也可以让 PyTorch 自动推断一维：

```python
y = x.reshape(3, -1)
```

因为总共有 12 个元素，所以 `-1` 自动推断为 4。

## 2. 增加维度

```python
x = torch.tensor([1.0, 2.0, 3.0])

y = x.unsqueeze(0)
z = x.unsqueeze(1)

print(x.shape)
print(y.shape)
print(z.shape)
```

输出：

```
torch.Size([3])
torch.Size([1, 3])
torch.Size([3, 1])
```

## 3. 删除长度为 1 的维度

```python
x = torch.randn(1, 3, 1)

y = x.squeeze()

print(x.shape)
print(y.shape)
```

输出：

```
torch.Size([1, 3, 1])
torch.Size([3])
```

## 4. 转置

```python
x = torch.randn(2, 3)
y = x.T

print(x.shape)
print(y.shape)
```

输出：

```
torch.Size([2, 3])
torch.Size([3, 2])
```

---

# 八、广播机制

```python
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

b = torch.tensor([10.0, 20.0, 30.0])

print(x + b)
```

输出：

```
tensor([[11., 22., 33.],
        [14., 25., 36.]])
```

虽然：

```python
x.shape == (2, 3)
b.shape == (3,)
```

但 PyTorch 会把 `b` 自动看成：

```
[[10, 20, 30],
 [10, 20, 30]]
```

然后执行运算。

这就是广播。

---

# 九、自动求导 autograd

这是 PyTorch 最关键的能力之一。

假设：

$$
y=x^2+3x
$$

那么：

$$
\frac{dy}{dx}=2x+3
$$

用 PyTorch：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x**2 + 3*x

y.backward()

print(x.grad)
```

输出：

```
tensor(7.)
```

因为：

$$
2\times2+3=7
$$

这里：

```python
requires_grad=True
```

表示：

> 请记录与这个 Tensor 有关的计算过程，以便之后求导。

```python
y.backward()
```

表示：

> 从 `y` 开始，沿计算图反向传播，计算所有需要的梯度。

```python
x.grad
```

保存：

$$
\frac{\partial y}{\partial x}
$$

PyTorch 的自动求导会记录 Tensor 运算并计算梯度。

---

## 梯度为什么要清零

看下面：

```python
x = torch.tensor(2.0, requires_grad=True)

y = x**2
y.backward()
print(x.grad)

y = x**2
y.backward()
print(x.grad)
```

输出：

```
tensor(4.)
tensor(8.)
```

因为 PyTorch 默认会**累加梯度**。

所以训练中每一轮通常都需要：

```python
optimizer.zero_grad()
```

否则这一轮的梯度会和上一轮叠加。

---

# 十、建立一个神经网络

PyTorch 中的模型通常继承：

```python
torch.nn.Module
```

官方推荐在 `__init__` 中定义网络层，在 `forward` 中定义数据如何经过网络。

```python
import torch
from torch import nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(2, 8)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = SimpleNet()
print(model)
```

输出大致是：

```
SimpleNet(
  (linear1): Linear(in_features=2, out_features=8, bias=True)
  (relu): ReLU()
  (linear2): Linear(in_features=8, out_features=1, bias=True)
)
```

---

## 逐项理解

### 1. 继承 nn.Module

```python
class SimpleNet(nn.Module):
```

表示：

> 我定义的 `SimpleNet` 是一种 PyTorch 模型。

### 2. 初始化父类

```python
super().__init__()
```

让 `nn.Module` 帮助模型：

- 管理参数；
- 移动到 GPU；
- 保存和加载参数；
- 切换训练与推理状态。

### 3. 定义线性层

```python
self.linear1 = nn.Linear(2, 8)
```

表示：

- 输入每个样本有 2 个特征；
- 输出每个样本有 8 个特征。

数学上大致是：

$$
y=xW^T+b
$$

### 4. 定义前向传播

```python
def forward(self, x):
```

描述输入如何经过模型。

调用时通常不要直接写：

```python
model.forward(x)
```

而是写：

```python
output = model(x)
```

PyTorch 会自动调用 `forward()`。

---

# 十一、完整训练示例：线性回归

假设真实规律是：

$$
y=3x+2
$$

让模型自己学习参数 3 和 2。

```python
import torch
from torch import nn

torch.manual_seed(42)

# 1. 创建训练数据
x = torch.linspace(-2, 2, 100).reshape(-1, 1)
y = 3 * x + 2

# 加一点随机噪声
y = y + 0.2 * torch.randn_like(y)

# 2. 定义模型
model = nn.Linear(1, 1)

# 3. 定义损失函数
loss_fn = nn.MSELoss()

# 4. 定义优化器
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.05
)

# 5. 训练
for epoch in range(500):
    # 前向传播
    prediction = model(x)

    # 计算损失
    loss = loss_fn(prediction, y)

    # 清除上一轮梯度
    optimizer.zero_grad()

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    if epoch % 50 == 0:
        print(
            f"epoch={epoch:3d}, "
            f"loss={loss.item():.6f}"
        )

# 6. 查看最终参数
print("weight:", model.weight.item())
print("bias:", model.bias.item())
```

最后可能得到：

```
weight: 3.01
bias: 2.00
```

也就是模型学到了：

$$
y\approx3.01x+2.00
$$

---

# 十二、训练循环到底在做什么

PyTorch 的典型训练循环是：

```python
for x, y in dataloader:
    prediction = model(x)
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

可以拆成五步。

## 第一步：模型预测

```python
prediction = model(x)
```

把输入送入模型，得到预测值。

## 第二步：计算损失

```python
loss = loss_fn(prediction, y)
```

计算预测值和真实值相差多少。

## 第三步：清空旧梯度

```python
optimizer.zero_grad()
```

因为梯度默认会累加。

## 第四步：计算梯度

```python
loss.backward()
```

自动求出：

$$
\frac{\partial L}{\partial w}
$$

也就是损失函数对每个模型参数的导数。

## 第五步：更新参数

```python
optimizer.step()
```

例如梯度下降大致执行：

$$
w_{\text{new}}
=
w_{\text{old}}
-\eta\frac{\partial L}{\partial w}
$$

其中 $\eta$ 就是学习率 `lr`。

---

# 十三、损失函数

## 1. 回归问题

预测连续数字，例如房价、温度、力、位移。

```python
loss_fn = nn.MSELoss()
```

均方误差：

$$
L=\frac1n\sum_{i=1}^n(\hat y_i-y_i)^2
$$

## 2. 多分类问题

例如判断图片是：

- 猫；
- 狗；
- 鸟。

使用：

```python
loss_fn = nn.CrossEntropyLoss()
```

注意：模型最后一般**不要手动加 Softmax**。

正确：

```python
model = nn.Linear(128, 3)
loss_fn = nn.CrossEntropyLoss()

logits = model(x)
loss = loss_fn(logits, labels)
```

其中：

```python
logits.shape == (batch_size, 3)
labels.shape == (batch_size,)
labels.dtype == torch.int64
```

标签应该类似：

```python
labels = torch.tensor([0, 2, 1, 0])
```

不是 one-hot：

```
[[1, 0, 0],
 [0, 0, 1],
 ...]
```

## 3. 二分类问题

常用：

```python
loss_fn = nn.BCEWithLogitsLoss()
```

模型输出一个未经 sigmoid 的数字：

```python
logits = model(x)
loss = loss_fn(logits, labels)
```

预测时再使用：

```python
probability = torch.sigmoid(logits)
prediction = probability >= 0.5
```

---

# 十四、优化器

常见优化器：

## SGD

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

带动量：

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

## Adam

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

初学时可以粗略记忆：

- 简单实验：`Adam` 容易上手。
- 想理解梯度下降：从 `SGD` 开始。
- `lr` 太大：损失可能震荡甚至爆炸。
- `lr` 太小：训练速度很慢。

---

# 十五、Dataset 和 DataLoader

实际数据通常不能一次全部塞进模型，所以需要分批读取。

PyTorch 的 `DataLoader` 是一个可以遍历数据集的对象，并支持批处理、打乱和多进程加载。

## 1. 使用 TensorDataset

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

x = torch.randn(1000, 10)
y = torch.randint(0, 3, (1000,))

dataset = TensorDataset(x, y)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

训练时：

```python
for batch_x, batch_y in dataloader:
    print(batch_x.shape)
    print(batch_y.shape)
    break
```

输出：

```
torch.Size([32, 10])
torch.Size([32])
```

表示一次取 32 个样本。

---

## 2. 自定义 Dataset

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y
```

使用：

```python
dataset = MyDataset(x, y)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

其中：

```python
__len__()
```

告诉 PyTorch 数据集有多少条数据。

```python
__getitem__(index)
```

告诉 PyTorch：

> 给我第 `index` 条数据时，应该返回什么。

---

# 十六、使用 GPU

最常见写法：

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)
```

然后把模型和数据都移动到同一个设备：

```python
model = model.to(device)

x = x.to(device)
y = y.to(device)
```

完整形式：

```python
for x, y in dataloader:
    x = x.to(device)
    y = y.to(device)

    prediction = model(x)
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 一个重要原则

模型和输入必须在同一个设备上。

错误情况：

```python
model在cuda
x在cpu
```

会出现类似错误：

```
Expected all tensors to be on the same device
```

---

# 十七、训练模式与推理模式

## 训练模式

```python
model.train()
```

## 推理或验证模式

```python
model.eval()
```

`train()` 和 `eval()` 会影响：

- Dropout；
- BatchNorm。

推理时还应该关闭梯度计算：

```python
model.eval()

with torch.no_grad():
    prediction = model(x)
```

更现代的推理写法是：

```python
model.eval()

with torch.inference_mode():
    prediction = model(x)
```

这样可以：

- 减少显存占用；
- 提高推理速度；
- 防止意外构建计算图。

---

# 十八、完整分类训练模板

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# -------------------------
# 1. 基本设置
# -------------------------

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)

# -------------------------
# 2. 模拟数据
# -------------------------

# 1000 个样本，每个样本 10 个特征
x = torch.randn(1000, 10)

# 三分类标签：0、1、2
y = torch.randint(0, 3, (1000,))

# -------------------------
# 3. 数据加载器
# -------------------------

dataset = TensorDataset(x, y)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# -------------------------
# 4. 定义模型
# -------------------------

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.network(x)

model = Classifier().to(device)

# -------------------------
# 5. 损失函数和优化器
# -------------------------

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# -------------------------
# 6. 训练
# -------------------------

epochs = 10

for epoch in range(epochs):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)

        loss = loss_fn(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

        predictions = logits.argmax(dim=1)

        total_correct += (
            predictions == batch_y
        ).sum().item()

        total_samples += batch_x.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    print(
        f"epoch={epoch + 1:2d}, "
        f"loss={average_loss:.4f}, "
        f"accuracy={accuracy:.2%}"
    )
```

---

# 十九、为什么 `argmax(dim=1)` 能得到类别

假设模型输出：

```python
logits = torch.tensor([
    [1.2, 3.5, 0.4],
    [4.1, 1.2, 2.3]
])
```

形状是：

```
(2, 3)
```

表示：

- 2 个样本；
- 每个样本有 3 个类别分数。

```python
predictions = logits.argmax(dim=1)
```

就是在每一行找最大值的位置：

```
tensor([1, 0])
```

表示：

- 第一个样本预测为类别 1；
- 第二个样本预测为类别 0。

---

# 二十、保存和加载模型

官方推荐保存模型的 `state_dict`，也就是参数字典。

## 1. 保存模型参数

```python
torch.save(
    model.state_dict(),
    "model_weights.pth"
)
```

## 2. 加载模型参数

必须先建立同样结构的模型：

```python
model = Classifier()

state_dict = torch.load(
    "model_weights.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()
```

`map_location="cpu"` 表示：

> 即使模型原来在 GPU 上保存，也先加载到 CPU。

这也可以避免加载时突然占用 GPU 显存。

---

## 3. 保存训练检查点

如果之后要继续训练，不能只保存模型，还要保存优化器状态和训练轮数。

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item()
}

torch.save(checkpoint, "checkpoint.pth")
```

加载：

```python
checkpoint = torch.load(
    "checkpoint.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

start_epoch = checkpoint["epoch"] + 1
```

---

# 二十一、PyTorch 和 NumPy 相互转换

```python
import numpy as np
import torch

array = np.array([1, 2, 3])

tensor = torch.from_numpy(array)

print(tensor)
```

Tensor 转 NumPy：

```python
array = tensor.numpy()
```

GPU Tensor 不能直接 `.numpy()`，需要：

```python
array = tensor.detach().cpu().numpy()
```

分别表示：

- `detach()`：脱离计算图；
- `cpu()`：移动到 CPU；
- `numpy()`：转换成 NumPy 数组。

---

# 二十二、常用神经网络层

## 全连接层

```python
nn.Linear(输入特征数, 输出特征数)
```

例如：

```python
nn.Linear(128, 64)
```

## 激活函数

```python
nn.ReLU()
nn.GELU()
nn.Sigmoid()
nn.Tanh()
```

现代 Transformer 中常见：

```python
nn.GELU()
```

## 卷积层

```python
nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3
)
```

## 池化层

```python
nn.MaxPool2d(kernel_size=2)
```

## Dropout

```python
nn.Dropout(p=0.5)
```

训练时随机关闭一部分神经元，降低过拟合。

## Embedding

```python
nn.Embedding(
    num_embeddings=10000,
    embedding_dim=256
)
```

把 token 编号转换为向量，是 NLP 和 LLM 的基础组件。

## Sequential

```python
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
)
```

适合按顺序连接网络层。

---

# 二十三、查看模型参数

```python
for name, parameter in model.named_parameters():
    print(name, parameter.shape)
```

可能输出：

```
network.0.weight torch.Size([64, 10])
network.0.bias torch.Size([64])
network.2.weight torch.Size([32, 64])
network.2.bias torch.Size([32])
network.4.weight torch.Size([3, 32])
network.4.bias torch.Size([3])
```

统计参数总数：

```python
total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(total_params)
```

只统计需要训练的参数：

```python
trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(trainable_params)
```

---

# 二十四、随机种子

```python
torch.manual_seed(42)
```

可以让随机初始化和随机数据尽量可复现。官方说明 `torch.manual_seed()` 会为 CPU 和 CUDA 设备设置 PyTorch 随机数种子。

但需要注意：

> 设置随机种子不保证所有平台、所有 GPU 算法、所有 PyTorch 版本的结果完全一致。

---

# 二十五、常见报错

## 1. 数据类型不匹配

报错类似：

```
mat1 and mat2 must have the same dtype
```

检查：

```python
print(x.dtype)
print(next(model.parameters()).dtype)
```

通常把输入改成：

```python
x = x.float()
```

---

## 2. 形状不匹配

报错类似：

```
mat1 and mat2 shapes cannot be multiplied
```

假设模型是：

```python
nn.Linear(10, 64)
```

那么输入最后一维必须是 10：

```python
x.shape == (batch_size, 10)
```

先打印：

```python
print(x.shape)
```

---

## 3. 模型和数据不在同一设备

报错：

```
Expected all tensors to be on the same device
```

处理：

```python
model = model.to(device)
x = x.to(device)
y = y.to(device)
```

---

## 4. CrossEntropyLoss 标签类型错误

错误写法：

```python
labels = labels.float()
```

正确写法：

```python
labels = labels.long()
```

因为 `CrossEntropyLoss` 的类别标签一般需要整数索引。

---

## 5. CUDA 显存不足

报错：

```
CUDA out of memory
```

优先尝试：

```python
batch_size = 16
```

改小为：

```python
batch_size = 8
```

并确认验证时使用：

```python
with torch.inference_mode():
    ...
```

还可以删除无用变量：

```python
del large_tensor
torch.cuda.empty_cache()
```

不过 `empty_cache()` 不能释放仍然被变量引用的 Tensor。

---

## 6. 忘记清零梯度

错误：

```python
loss.backward()
optimizer.step()
```

训练多轮后梯度不断累积。

通常应该：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

# 二十六、建议的学习顺序

第一阶段先掌握：

```
Tensor 创建
Tensor 形状
索引切片
reshape
矩阵乘法
dim 参数
CPU/GPU 转移
```

第二阶段掌握：

```
requires_grad
loss.backward()
parameter.grad
计算图
梯度清零
```

第三阶段掌握：

```
nn.Module
nn.Linear
激活函数
损失函数
优化器
标准训练循环
```

第四阶段掌握：

```
Dataset
DataLoader
训练集和验证集
model.train()
model.eval()
保存和加载
```

第五阶段再进入：

```
CNN
RNN
Attention
Transformer
混合精度训练
多 GPU
LLM 微调
```

---

# 二十七、你目前最应该记住的模板

```python
import torch
from torch import nn

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(10):
    model.train()

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()

with torch.inference_mode():
    logits = model(test_x.to(device))
    predictions = logits.argmax(dim=1)
```

先不要急着把它完全背下来。只需要理解这条主线：

$$
\boxed{
数据
\rightarrow 模型预测
\rightarrow 计算损失
\rightarrow 反向传播
\rightarrow 更新参数
}
$$

PyTorch 的绝大多数模型，无论是简单的线性回归、CNN，还是 Transformer 和 LLM，训练过程的核心结构都没有脱离这条主线。
