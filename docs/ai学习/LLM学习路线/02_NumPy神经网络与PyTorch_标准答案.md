# 模块 2 标准答案：NumPy 神经网络与 PyTorch

对应文件：[02_NumPy神经网络与PyTorch.md](02_NumPy神经网络与PyTorch.md)

本文件有两类内容：前半部分是原文讲解中穿插的手算与代码示例的完整解答；后半部分是两个课后 Project 的完整参考实现。

建议使用方式：先在原文相应位置自己推导或编写；卡住时回看讲解；完成后再对照这里的完整答案。课后作业请优先查看后半部分的 Project 1 和 Project 2。

---

## 示例 1 完整解答：单神经元与线性层手算

题目给定：

$$
x_1=3,\quad x_2=1,\quad w_1=0.8,\quad w_2=-1.2,\quad b=-0.5
$$

计算：

$$
z=w_1x_1+w_2x_2+b
$$

代入：

$$
z=0.8\times3+(-1.2)\times1-0.5
$$

$$
z=2.4-1.2-0.5=0.7
$$

答案：

$$
\boxed{z=0.7}
$$

解释：

- 正权重表示该特征会把分数往上推。
- 负权重表示该特征会把分数往下拉。
- 偏置 $b$ 表示模型在没有输入特征贡献时的基础倾向。
- 单个神经元本质上就是对输入特征做加权求和，再加一个偏置。

---

## 示例 2 完整解答：批量线性层形状推导

题目给定：

$$
X:[B,D_{\text{in}}],
\quad
W:[D_{\text{in}},D_{\text{out}}],
\quad
b:[D_{\text{out}}]
$$

线性层：

$$
Z=XW+b
$$

矩阵乘法部分：

$$
[B,D_{\text{in}}] @ [D_{\text{in}},D_{\text{out}}]
=
[B,D_{\text{out}}]
$$

偏置 $b$ 的形状是：

$$
[D_{\text{out}}]
$$

它会沿 batch 维广播，相当于：

$$
[1,D_{\text{out}}]
\rightarrow
[B,D_{\text{out}}]
$$

所以最终：

$$
\boxed{Z:[B,D_{\text{out}}]}
$$

手算例子：

$$
X=
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
$$

$$
W=
\begin{bmatrix}
0.1 & -0.2\\
0.3 & 0.4
\end{bmatrix}
$$

$$
b=
\begin{bmatrix}
0.01 & -0.02
\end{bmatrix}
$$

先算：

$$
XW=
\begin{bmatrix}
1\times0.1+2\times0.3 & 1\times(-0.2)+2\times0.4\\
3\times0.1+4\times0.3 & 3\times(-0.2)+4\times0.4
\end{bmatrix}
$$

$$
XW=
\begin{bmatrix}
0.7 & 0.6\\
1.5 & 1.0
\end{bmatrix}
$$

加偏置：

$$
Z=
\begin{bmatrix}
0.7 & 0.6\\
1.5 & 1.0
\end{bmatrix}
+
\begin{bmatrix}
0.01 & -0.02
\end{bmatrix}
$$

$$
\boxed{
Z=
\begin{bmatrix}
0.71 & 0.58\\
1.51 & 0.98
\end{bmatrix}
}
$$

---

## 示例 3 完整解答：Softmax 与交叉熵

给定：

```python
z = [2.0, 1.0, 0.1]
```

softmax：

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

近似结果：

```python
p = [0.6590, 0.2424, 0.0986]
```

如果正确类别是第 0 类：

$$
L=-\log p_0
$$

$$
L=-\log(0.6590)\approx0.417
$$

答案：

$$
\boxed{L\approx0.417}
$$

稳定 softmax 代码：

```python
import numpy as np

def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)

z = np.array([2.0, 1.0, 0.1])
p = softmax(z)
loss = -np.log(p[0])

print(p)
print(loss)
```

为什么减最大值：

$$
\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}
=
\frac{e^{z_i}/e^m}{\sum_j e^{z_j}/e^m}
=
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

所以结果不变，但能避免 `np.exp(1000)` 这种指数溢出。

为什么 PyTorch 的 `CrossEntropyLoss` 不提前 softmax：

`nn.CrossEntropyLoss` 内部已经做了稳定版：

```text
log_softmax + NLLLoss
```

所以正确输入是原始 logits：

```python
loss = criterion(logits, targets)
```

不要写成：

```python
loss = criterion(torch.softmax(logits, dim=-1), targets)
```

---

## 示例 4 完整解答：多层 MLP 前向传播

三层 MLP：

```text
X
→ Linear1
→ ReLU
→ Linear2
→ ReLU
→ Linear3
→ logits
```

设：

$$
A_0=X
$$

前向传播：

$$
Z_1=A_0W_1+b_1
$$

$$
A_1=ReLU(Z_1)
$$

$$
Z_2=A_1W_2+b_2
$$

$$
A_2=ReLU(Z_2)
$$

$$
Z_3=A_2W_3+b_3
$$

$$
logits=Z_3
$$

形状表：

| 变量 | 形状 |
|---|---:|
| $A_0=X$ | $[B,D_{\text{in}}]$ |
| $W_1$ | $[D_{\text{in}},H_1]$ |
| $b_1$ | $[H_1]$ |
| $Z_1,A_1$ | $[B,H_1]$ |
| $W_2$ | $[H_1,H_2]$ |
| $b_2$ | $[H_2]$ |
| $Z_2,A_2$ | $[B,H_2]$ |
| $W_3$ | $[H_2,C]$ |
| $b_3$ | $[C]$ |
| $Z_3$ | $[B,C]$ |

为什么中间层要保存 cache：

- `Linear.backward()` 需要前向输入 $A_{\ell-1}$ 来计算 $A_{\ell-1}^\mathsf{T}G_\ell$。
- `ReLU.backward()` 需要前向时的 $Z_\ell$ 或 mask 来判断哪些位置梯度通过。

为什么最后输出 logits：

- 分类任务最后交给交叉熵。
- `CrossEntropyLoss` 需要原始 logits。
- 如果提前 softmax，数值稳定性和梯度都会变差。

---

## 示例 5 完整解答：多层 MLP 反向传播

假设：

$$
G_3=\frac{\partial L}{\partial Z_3}
$$

输出层：

$$
\frac{\partial L}{\partial W_3}=A_2^\mathsf{T}G_3
$$

$$
\frac{\partial L}{\partial b_3}=\operatorname{sum}(G_3,\text{axis}=0)
$$

$$
\frac{\partial L}{\partial A_2}=G_3W_3^\mathsf{T}
$$

ReLU2：

$$
G_2
=
\frac{\partial L}{\partial Z_2}
=
\frac{\partial L}{\partial A_2}
\odot
\mathbf{1}[Z_2>0]
$$

第二层：

$$
\frac{\partial L}{\partial W_2}=A_1^\mathsf{T}G_2
$$

$$
\frac{\partial L}{\partial b_2}=\operatorname{sum}(G_2,\text{axis}=0)
$$

$$
\frac{\partial L}{\partial A_1}=G_2W_2^\mathsf{T}
$$

ReLU1：

$$
G_1
=
\frac{\partial L}{\partial Z_1}
=
\frac{\partial L}{\partial A_1}
\odot
\mathbf{1}[Z_1>0]
$$

第一层：

$$
\frac{\partial L}{\partial W_1}=A_0^\mathsf{T}G_1
$$

$$
\frac{\partial L}{\partial b_1}=\operatorname{sum}(G_1,\text{axis}=0)
$$

$$
\frac{\partial L}{\partial A_0}=G_1W_1^\mathsf{T}
$$

最关键的回答：

> 上一层输出如何影响最终 loss，是由后一层传回来的梯度和后一层权重共同决定的。

例如：

$$
\frac{\partial L}{\partial A_2}=G_3W_3^\mathsf{T}
$$

这不是猜出来的，是链式法则算出来的。

---

## Project 1 标准答案：NumPy MLP 完整实现

文件建议保存为：

```text
module02_homework/numpy_mlp_solution.py
```

完整代码：

```python
import numpy as np


class Linear:
    def __init__(self, in_features, out_features, rng):
        self.W = rng.normal(
            0.0,
            np.sqrt(2.0 / in_features),
            size=(in_features, out_features),
        )
        self.b = np.zeros(out_features)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        self.grad_W = self.x.T @ grad_out
        self.grad_b = grad_out.sum(axis=0)
        grad_x = grad_out @ self.W.T
        return grad_x


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, grad_out):
        return grad_out * self.mask


class SoftmaxCrossEntropy:
    def forward(self, logits, targets):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / exp.sum(axis=1, keepdims=True)
        self.targets = targets

        n = logits.shape[0]
        correct_probs = self.probs[np.arange(n), targets]
        loss = -np.log(correct_probs + 1e-12).mean()
        return loss

    def backward(self):
        n = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(n), self.targets] -= 1
        return grad / n


class MLP:
    def __init__(self, input_dim, hidden_dims, num_classes, rng):
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        self.layers = []

        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1], rng))
            if i < len(dims) - 2:
                self.layers.append(ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        for layer in self.layers:
            if isinstance(layer, Linear):
                yield layer


def sgd_step(model, lr):
    for layer in model.parameters():
        layer.W -= lr * layer.grad_W
        layer.b -= lr * layer.grad_b


def make_blobs(n_per_class=100, seed=42):
    rng = np.random.default_rng(seed)

    x0 = rng.normal(loc=(-1.5, -1.5), scale=0.7, size=(n_per_class, 2))
    y0 = np.zeros(n_per_class, dtype=np.int64)

    x1 = rng.normal(loc=(1.5, 1.5), scale=0.7, size=(n_per_class, 2))
    y1 = np.ones(n_per_class, dtype=np.int64)

    X = np.vstack([x0, x1])
    y = np.concatenate([y0, y1])

    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def accuracy(logits, y):
    pred = logits.argmax(axis=1)
    return np.mean(pred == y)


def train():
    rng = np.random.default_rng(0)
    X, y = make_blobs(n_per_class=200)

    split = int(0.8 * len(y))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = MLP(
        input_dim=2,
        hidden_dims=[32, 32],
        num_classes=2,
        rng=rng,
    )
    criterion = SoftmaxCrossEntropy()

    lr = 0.05
    batch_size = 32
    epochs = 50

    for epoch in range(epochs):
        idx = rng.permutation(len(y_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

        total_loss = 0.0
        total_count = 0

        for start in range(0, len(y_train), batch_size):
            end = start + batch_size
            xb = X_train[start:end]
            yb = y_train[start:end]

            logits = model.forward(xb)
            loss = criterion.forward(logits, yb)
            grad = criterion.backward()
            model.backward(grad)
            sgd_step(model, lr)

            total_loss += loss * len(yb)
            total_count += len(yb)

        train_logits = model.forward(X_train)
        val_logits = model.forward(X_val)

        train_acc = accuracy(train_logits, y_train)
        val_acc = accuracy(val_logits, y_val)
        avg_loss = total_loss / total_count

        if epoch % 5 == 0:
            print(
                f"epoch={epoch:03d} "
                f"loss={avg_loss:.4f} "
                f"train_acc={train_acc:.3f} "
                f"val_acc={val_acc:.3f}"
            )


if __name__ == "__main__":
    train()
```

形状表：

| 层 | forward 输入 | forward 输出 | backward 输入 | backward 输出 |
|---|---:|---:|---:|---:|
| `Linear(2,32)` | `[B,2]` | `[B,32]` | `[B,32]` | `[B,2]` |
| `ReLU` | `[B,32]` | `[B,32]` | `[B,32]` | `[B,32]` |
| `Linear(32,32)` | `[B,32]` | `[B,32]` | `[B,32]` | `[B,32]` |
| `ReLU` | `[B,32]` | `[B,32]` | `[B,32]` | `[B,32]` |
| `Linear(32,2)` | `[B,32]` | `[B,2]` | `[B,2]` | `[B,32]` |

最小梯度检查示例：

```python
def numerical_gradient_check():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(5, 2))
    y = np.array([0, 1, 0, 1, 1])

    model = MLP(2, [4], 2, rng)
    criterion = SoftmaxCrossEntropy()

    logits = model.forward(X)
    loss = criterion.forward(logits, y)
    grad = criterion.backward()
    model.backward(grad)

    first_linear = next(model.parameters())
    i, j = 0, 0
    analytic = first_linear.grad_W[i, j]

    eps = 1e-5
    old = first_linear.W[i, j]

    first_linear.W[i, j] = old + eps
    loss_pos = criterion.forward(model.forward(X), y)

    first_linear.W[i, j] = old - eps
    loss_neg = criterion.forward(model.forward(X), y)

    first_linear.W[i, j] = old

    numerical = (loss_pos - loss_neg) / (2 * eps)
    rel_error = abs(analytic - numerical) / max(
        1e-8,
        abs(analytic) + abs(numerical),
    )

    print("analytic:", analytic)
    print("numerical:", numerical)
    print("relative error:", rel_error)
```

---

## Project 2 标准答案：PyTorch MLP 项目

### Project 2.1 环境准备

先按 [PyTorch 使用指南](../PyTorch使用指南.md) 创建独立 Conda 环境、安装 CPU 或 NVIDIA GPU 版 PyTorch，并完成其中的导入与 Tensor 计算检查。确认 `import torch` 没有报错后，再运行本项目代码。

### Project 2.2 PyTorch MNIST/FashionMNIST 完整训练脚本

文件建议保存为：

```text
module02_homework/pytorch_mnist_mlp.py
```

完整代码：

```python
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = x.flatten(1)
        return self.net(x)


def make_loaders(data_dir, batch_size, dataset_name):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if dataset_name == "mnist":
        dataset_cls = datasets.MNIST
    elif dataset_name == "fashion":
        dataset_cls = datasets.FashionMNIST
    else:
        raise ValueError("dataset_name must be 'mnist' or 'fashion'")

    train_set = dataset_cls(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_set = dataset_cls(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


def run_one_epoch(model, loader, criterion, optimizer, device, train):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += batch_size

    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
    }


def save_checkpoint(path, model, optimizer, epoch, best_acc):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="fashion")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, default="mlp_checkpoint.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_loader, test_loader = make_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        dataset_name=args.dataset,
    )

    model = MLP(hidden_dim=args.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
        )
        test_metrics = run_one_epoch(
            model,
            test_loader,
            criterion,
            optimizer,
            device,
            train=False,
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f}"
        )

        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            save_checkpoint(
                args.ckpt,
                model,
                optimizer,
                epoch,
                best_acc,
            )

    print("best_acc:", best_acc)

    # 保存加载一致性检查
    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        x, _ = next(iter(test_loader))
        x = x.to(device)

        model.eval()
        with torch.no_grad():
            logits_before = model(x[:4])

        loaded_model = MLP(hidden_dim=args.hidden_dim).to(device)
        load_checkpoint(args.ckpt, loaded_model, device=device)
        loaded_model.eval()

        with torch.no_grad():
            logits_after = loaded_model(x[:4])

        print(
            "reload logits allclose:",
            torch.allclose(logits_before, logits_after, atol=1e-6),
        )


if __name__ == "__main__":
    main()
```

运行示例：

```bash
python pytorch_mnist_mlp.py --dataset fashion --optimizer adam --hidden-dim 256 --epochs 5
python pytorch_mnist_mlp.py --dataset fashion --optimizer sgd --hidden-dim 256 --epochs 5 --lr 0.1
python pytorch_mnist_mlp.py --dataset fashion --optimizer adam --hidden-dim 64 --epochs 5
python pytorch_mnist_mlp.py --dataset fashion --optimizer adam --hidden-dim 1024 --epochs 5
```

实验结论应该包含：

- Adam 通常前期收敛更快。
- SGD 对学习率更敏感。
- 隐藏宽度更大通常表达能力更强，但参数更多、速度更慢，也更容易过拟合。
- `model.eval()` 用于切换模型行为，`torch.no_grad()` 用于关闭梯度记录。
- 保存加载后，同一输入的 logits 应该在容差内一致。
