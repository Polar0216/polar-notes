# 模块 2：NumPy 与 PyTorch MLP 完整模型代码

对应主讲义：[02_NumPy神经网络与PyTorch.md](02_NumPy神经网络与PyTorch.md)

这份文件只解决一个问题：把主讲义中分开讲解的组件拼成**可以直接保存和运行的完整程序**。建议先学习主讲义，再按下面顺序使用代码：

1. 先运行原版，确认环境和数据流程正常。
2. 从 `main()` 开始阅读，再依次看训练循环、模型、组件。
3. 独立修改一项配置，例如隐藏层宽度或优化器。
4. 最后回到主讲义第 22 节“课后综合项目”，尝试不看本文件重新实现。

---

## 1. NumPy MLP：从零实现完整训练流程

### 1.1 文件保存位置与运行方法

建议保存为：

```text
module02_code/
└── numpy_mlp_complete.py
```

依赖：

```powershell
python -m pip install numpy
```

运行：

```powershell
python numpy_mlp_complete.py
```

### 1.2 完整代码

```python
from __future__ import annotations

from pathlib import Path

import numpy as np


class Linear:
    """全连接层：Z = XW + b。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator,
    ) -> None:
        # He 初始化适合后面接 ReLU。
        scale = np.sqrt(2.0 / in_features)
        self.W = rng.normal(
            loc=0.0,
            scale=scale,
            size=(in_features, out_features),
        )
        self.b = np.zeros(out_features, dtype=np.float64)

        self.x: np.ndarray | None = None
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: [B, in_features]
        # W: [in_features, out_features]
        # b: [out_features]，沿 batch 维广播。
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("必须先调用 forward()，再调用 backward()")

        # grad_out: [B, out_features]
        self.grad_W = self.x.T @ grad_out
        self.grad_b = grad_out.sum(axis=0)
        grad_x = grad_out @ self.W.T
        return grad_x


class ReLU:
    """ReLU 激活层。"""

    def __init__(self) -> None:
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise RuntimeError("必须先调用 forward()，再调用 backward()")
        return grad_out * self.mask


class SoftmaxCrossEntropy:
    """数值稳定的 Softmax 与多分类交叉熵。"""

    def __init__(self) -> None:
        self.probs: np.ndarray | None = None
        self.targets: np.ndarray | None = None

    def forward(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        # 每个样本减去自己的最大 logit。
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)

        self.probs = probs
        self.targets = targets

        batch_size = logits.shape[0]
        correct_probs = probs[np.arange(batch_size), targets]
        loss = -np.log(correct_probs + 1e-12).mean()
        return float(loss)

    def backward(self) -> np.ndarray:
        if self.probs is None or self.targets is None:
            raise RuntimeError("必须先调用 forward()，再调用 backward()")

        batch_size = self.probs.shape[0]
        grad_logits = self.probs.copy()
        grad_logits[np.arange(batch_size), self.targets] -= 1.0
        grad_logits /= batch_size
        return grad_logits


class MLP:
    """由多个 Linear 与 ReLU 组成的多层感知机。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        rng: np.random.Generator,
    ) -> None:
        dims = [input_dim, *hidden_dims, num_classes]
        self.layers: list[Linear | ReLU] = []

        for index in range(len(dims) - 1):
            self.layers.append(
                Linear(dims[index], dims[index + 1], rng)
            )

            # 输出层保留原始 logits，不接 ReLU。
            if index < len(dims) - 2:
                self.layers.append(ReLU())

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def linear_layers(self) -> list[Linear]:
        return [
            layer
            for layer in self.layers
            if isinstance(layer, Linear)
        ]

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        return logits.argmax(axis=1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {}
        for index, layer in enumerate(self.linear_layers()):
            arrays[f"W{index}"] = layer.W
            arrays[f"b{index}"] = layer.b

        np.savez(path, **arrays)

    def load(self, path: str | Path) -> None:
        data = np.load(path)
        for index, layer in enumerate(self.linear_layers()):
            layer.W[...] = data[f"W{index}"]
            layer.b[...] = data[f"b{index}"]


def make_spiral_dataset(
    samples_per_class: int = 200,
    num_classes: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """生成非线性的二维螺旋分类数据。"""

    rng = np.random.default_rng(seed)
    total = samples_per_class * num_classes
    x = np.zeros((total, 2), dtype=np.float64)
    y = np.zeros(total, dtype=np.int64)

    for class_id in range(num_classes):
        start = class_id * samples_per_class
        end = start + samples_per_class

        radius = np.linspace(0.05, 1.0, samples_per_class)
        angle = np.linspace(
            class_id * 4.0,
            (class_id + 1) * 4.0,
            samples_per_class,
        )
        angle += rng.normal(0.0, 0.25, samples_per_class)

        x[start:end, 0] = radius * np.sin(angle)
        x[start:end, 1] = radius * np.cos(angle)
        y[start:end] = class_id

    order = rng.permutation(total)
    return x[order], y[order]


def split_and_normalize(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = int(len(x) * train_ratio)
    x_train = x[:split]
    y_train = y[:split]
    x_val = x[split:]
    y_val = y[split:]

    # 只能用训练集统计量，避免验证集信息泄漏。
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-8

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    return x_train, y_train, x_val, y_val


def accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    predictions = logits.argmax(axis=1)
    return float((predictions == targets).mean())


def sgd_step(model: MLP, learning_rate: float) -> None:
    for layer in model.linear_layers():
        layer.W -= learning_rate * layer.grad_W
        layer.b -= learning_rate * layer.grad_b


def evaluate(
    model: MLP,
    loss_fn: SoftmaxCrossEntropy,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    logits = model.forward(x)
    loss = loss_fn.forward(logits, y)
    return loss, accuracy(logits, y)


def gradient_check(
    model: MLP,
    loss_fn: SoftmaxCrossEntropy,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-5,
) -> float:
    """检查第一个权重的解析梯度与数值梯度。"""

    first_linear = model.linear_layers()[0]
    index = (0, 0)

    logits = model.forward(x)
    loss_fn.forward(logits, y)
    model.backward(loss_fn.backward())
    analytic = float(first_linear.grad_W[index])

    original = float(first_linear.W[index])

    first_linear.W[index] = original + epsilon
    loss_plus = loss_fn.forward(model.forward(x), y)

    first_linear.W[index] = original - epsilon
    loss_minus = loss_fn.forward(model.forward(x), y)

    first_linear.W[index] = original
    numerical = (loss_plus - loss_minus) / (2.0 * epsilon)

    denominator = max(
        1e-12,
        abs(analytic) + abs(numerical),
    )
    relative_error = abs(analytic - numerical) / denominator

    print("gradient check analytic :", analytic)
    print("gradient check numerical:", numerical)
    print("gradient check rel error:", relative_error)
    return relative_error


def train(
    model: MLP,
    loss_fn: SoftmaxCrossEntropy,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 2000,
    learning_rate: float = 0.5,
) -> None:
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # 1. 前向
        train_logits = model.forward(x_train)

        # 2. 损失
        train_loss = loss_fn.forward(train_logits, y_train)

        # 3. 反向
        grad_logits = loss_fn.backward()
        model.backward(grad_logits)

        # 4. 更新
        sgd_step(model, learning_rate)

        if epoch == 1 or epoch % 100 == 0:
            train_acc = accuracy(train_logits, y_train)
            val_loss, val_acc = evaluate(
                model,
                loss_fn,
                x_val,
                y_val,
            )

            print(
                f"epoch={epoch:4d} "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.3f} "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.3f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save("outputs/numpy_mlp_best.npz")


def main() -> None:
    seed = 42
    rng = np.random.default_rng(seed)

    x, y = make_spiral_dataset(
        samples_per_class=200,
        num_classes=3,
        seed=seed,
    )
    x_train, y_train, x_val, y_val = split_and_normalize(x, y)

    model = MLP(
        input_dim=2,
        hidden_dims=[64, 64],
        num_classes=3,
        rng=rng,
    )
    loss_fn = SoftmaxCrossEntropy()

    relative_error = gradient_check(
        model,
        loss_fn,
        x_train[:8],
        y_train[:8],
    )
    if relative_error > 1e-5:
        raise RuntimeError("梯度检查失败，请先检查 backward 实现")

    train(
        model=model,
        loss_fn=loss_fn,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=2000,
        learning_rate=0.5,
    )

    model.load("outputs/numpy_mlp_best.npz")
    val_loss, val_acc = evaluate(
        model,
        loss_fn,
        x_val,
        y_val,
    )
    print(f"best checkpoint: loss={val_loss:.4f}, acc={val_acc:.3f}")


if __name__ == "__main__":
    main()
```

### 1.3 NumPy MLP 的完整数据流

```text
二维数据 X
  → MLP.forward()
  → logits
  → SoftmaxCrossEntropy.forward()
  → loss
  → SoftmaxCrossEntropy.backward()
  → grad_logits
  → MLP.backward()
  → 每个 Linear 的 grad_W、grad_b
  → sgd_step()
  → 更新后的 W、b
```

---

## 2. PyTorch MLP：MNIST/FashionMNIST 完整训练流程

### 2.1 环境准备与运行方法

环境安装请先完成：[PyTorch 使用指南](../PyTorch使用指南.md)

建议保存为：

```text
module02_code/
└── pytorch_mlp_complete.py
```

CPU 运行示例：

```powershell
conda activate pytorch-study
python pytorch_mlp_complete.py --dataset fashion_mnist --epochs 10
```

使用 SGD 和不同隐藏宽度：

```powershell
python pytorch_mlp_complete.py --optimizer sgd --hidden-dims 256,128
```

### 2.2 完整代码

```python
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 有助于复现，但不同硬件和算子仍可能存在细小差异。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dims: list[int] | None = None,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        dims = [input_dim, *hidden_dims, num_classes]
        layers: list[nn.Module] = [nn.Flatten()]

        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))

            # 最后一层输出 logits，不添加 ReLU。
            if index < len(dims) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.ToTensor()

    if dataset_name == "mnist":
        dataset_class = datasets.MNIST
    elif dataset_name == "fashion_mnist":
        dataset_class = datasets.FashionMNIST
    else:
        raise ValueError(f"未知数据集：{dataset_name}")

    full_train = dataset_class(
        root=data_dir,
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = dataset_class(
        root=data_dir,
        train=False,
        transform=transform,
        download=True,
    )

    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=generator,
    )

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, test_loader


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = (
        torch.enable_grad()
        if training
        else torch.inference_mode()
    )

    with context:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad()

            logits = model(x)
            loss = loss_fn(logits, y)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def build_optimizer(
    name: str,
    model: nn.Module,
    learning_rate: float,
) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
        )

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

    raise ValueError(f"未知优化器：{name}")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint


def parse_hidden_dims(text: str) -> list[int]:
    dims = [int(value.strip()) for value in text.split(",")]
    if not dims or any(value <= 0 for value in dims):
        raise argparse.ArgumentTypeError("隐藏层宽度必须是正整数")
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion_mnist"],
        default="fashion_mnist",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--hidden-dims",
        type=parse_hidden_dims,
        default=parse_hidden_dims("256,128"),
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam"],
        default="adam",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--checkpoint",
        default="outputs/pytorch_mlp_best.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    print("device:", device)

    train_loader, val_loader, test_loader = make_loaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )

    model = MLP(
        input_dim=28 * 28,
        hidden_dims=args.hidden_dims,
        num_classes=10,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        name=args.optimizer,
        model=model,
        learning_rate=args.learning_rate,
    )

    config = vars(args).copy()
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
        )
        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                path=args.checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                config=config,
            )

    checkpoint = load_checkpoint(
        path=args.checkpoint,
        model=model,
        optimizer=None,
        device=device,
    )
    test_metrics = run_one_epoch(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print("loaded epoch:", checkpoint["epoch"])
    print(
        f"test_loss={test_metrics['loss']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
```

### 2.3 PyTorch MLP 的完整数据流

```text
Dataset
  → DataLoader 组成 batch
  → 数据移动到 device
  → model(x) 得到 logits
  → CrossEntropyLoss(logits, y)
  → optimizer.zero_grad()
  → loss.backward()
  → optimizer.step()
  → 验证集评估
  → 保存最佳 checkpoint
  → 加载最佳 checkpoint
  → 测试集评估
```

---

## 3. NumPy 与 PyTorch 实现对照

| 问题 | NumPy 手写版 | PyTorch 版 |
|---|---|---|
| 权重存储 | `W:[D_in,D_out]` | `nn.Linear.weight:[D_out,D_in]` |
| 前向表达 | `x @ W + b` | 等价于 `x @ W.T + b` |
| 中间缓存 | 每层手动保存 | 计算图自动管理 |
| 清梯度 | 每次直接覆盖 `grad_W/grad_b` | 每个 step 调用 `optimizer.zero_grad()` |
| 反向传播 | 手动倒序调用 | `loss.backward()` |
| 参数更新 | 手写 `W -= lr * grad_W` | `optimizer.step()` |
| 评估模式 | 自己保证不反向 | `model.eval()` 与 `torch.inference_mode()` |

读代码时始终跟踪四件事：输入形状、输出形状、这一层保存了什么、反向时返回什么。只要这四件事明确，多层 MLP 就不会再像一堆零散 API。
