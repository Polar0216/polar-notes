# 模块 1 课后 Project 标准答案

对应文件：[01_Python_Git_数学基础.md](01_Python_Git_数学基础.md)

建议使用方式：

1. 先完成原文件文末的“课后 Project”。
2. 自己写一版，哪怕很粗糙。
3. 卡住时看原文件里的“解题提示与逐步讲解”。
4. 最后再用本文件对照完整答案。

---

## Project 1 标准答案：纯 Python 文本分析

文件建议保存为：

```text
module01_homework/exercise_a_jsonl_stats.py
```

完整代码：

```python
import json
from collections import Counter
from pathlib import Path


def get_text(obj: dict) -> str:
    """从一条 JSON 记录中提取文本。

    优先级：
    1. text
    2. instruction + output
    3. 空字符串
    """
    if "text" in obj:
        return str(obj["text"])

    if "instruction" in obj or "output" in obj:
        instruction = str(obj.get("instruction", ""))
        output = str(obj.get("output", ""))
        return f"{instruction} {output}".strip()

    return ""


def bucket_length(n: int) -> str:
    """把文本长度分到几个粗略区间。"""
    if n == 0:
        return "0"
    if n <= 10:
        return "1-10"
    if n <= 50:
        return "11-50"
    if n <= 100:
        return "51-100"
    return "100+"


def analyze_jsonl(path: Path) -> dict:
    total_lines = 0
    valid_records = 0
    bad_lines = 0
    empty_texts = 0
    duplicate_texts = 0

    seen_texts = set()
    length_buckets = Counter()
    char_counter = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            total_lines += 1
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                print(f"[坏行] line={line_no}: {line[:80]}")
                continue

            valid_records += 1
            text = get_text(obj).strip()

            if not text:
                empty_texts += 1
                continue

            if text in seen_texts:
                duplicate_texts += 1
            else:
                seen_texts.add(text)

            length_buckets[bucket_length(len(text))] += 1
            char_counter.update(text)

    return {
        "total_lines": total_lines,
        "valid_records": valid_records,
        "bad_lines": bad_lines,
        "empty_texts": empty_texts,
        "duplicate_texts": duplicate_texts,
        "length_buckets": dict(length_buckets),
        "top_chars": char_counter.most_common(20),
    }


def main():
    current_dir = Path(__file__).parent
    path = current_dir / "data" / "demo.jsonl"

    result = analyze_jsonl(path)

    print("\n========== JSONL 统计结果 ==========")
    print("总行数:", result["total_lines"])
    print("合法 JSON 记录数:", result["valid_records"])
    print("坏行数:", result["bad_lines"])
    print("空文本数:", result["empty_texts"])
    print("重复文本数:", result["duplicate_texts"])
    print("长度分桶:", result["length_buckets"])
    print("高频字符 Top 20:", result["top_chars"])


if __name__ == "__main__":
    main()
```

配套 demo 数据：

```text
module01_homework/data/demo.jsonl
```

内容：

```jsonl
{"text": "hello world"}
{"text": ""}
{"text": "机器学习"}
{"text": "hello world"}
这不是合法 JSON
{"instruction": "解释 AI", "output": "AI 是人工智能。"}
```

你应该观察到：

- 程序不会因为非法 JSON 行崩溃；
- 空文本会被统计；
- 重复的 `"hello world"` 会被统计；
- 没有 `text` 字段时，会尝试拼接 `instruction` 和 `output`。

---

## Project 2 标准答案：NumPy 形状体操

文件建议保存为：

```text
module01_homework/exercise_b_numpy_shapes.py
```

完整代码：

```python
import numpy as np


def softmax(x, axis=-1):
    """稳定版 softmax。"""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def main():
    rng = np.random.default_rng(42)

    B = 2
    T = 3
    C = 4
    V = 5

    # x: [B, T, C]
    x = rng.normal(size=(B, T, C))

    # W: [C, V]
    W = rng.normal(size=(C, V))

    print("x.shape =", x.shape)
    print("W.shape =", W.shape)

    # 1. 每个 token 在 C 维上的均值：[B, T, C] -> [B, T]
    token_mean = x.mean(axis=-1)
    print("token_mean.shape =", token_mean.shape)

    # 2. 展平最后两个维度：[B, T, C] -> [B, T*C]
    flat = x.reshape(B, T * C)
    print("flat.shape =", flat.shape)

    # 3. 线性变换：[B, T, C] @ [C, V] -> [B, T, V]
    logits = x @ W
    print("logits.shape =", logits.shape)

    # 4. 对 V 维做 softmax
    probs = softmax(logits, axis=-1)
    print("probs.shape =", probs.shape)

    # 5. 验证概率和：[B, T, V] -> [B, T]
    prob_sums = probs.sum(axis=-1)
    print("prob_sums.shape =", prob_sums.shape)
    print(prob_sums)
    print("all close to 1:", np.allclose(prob_sums, 1.0))


if __name__ == "__main__":
    main()
```

关键解释：

- `x.mean(axis=-1)` 会消掉最后的 `C` 维，所以 `[B,T,C]` 变成 `[B,T]`。
- `reshape(B, T*C)` 保留 batch 维，因为 batch 维表示样本数量，不应该和特征维混在一起。
- `x @ W` 会用 `x` 的最后一维 `C` 和 `W` 的第一维 `C` 做矩阵乘法。
- softmax 要在最后一维 `V` 上做，因为 `V` 表示每个类别或词表位置的分数。

---

## Project 3 标准答案：NumPy 线性回归

文件建议保存为：

```text
module01_homework/exercise_c_linear_regression.py
```

完整代码：

```python
import numpy as np


def make_data(seed=42, n=100):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, size=n)
    noise = rng.normal(0, 1.0, size=n)
    y = 3 * x + 2 + noise
    return x, y


def predict(x, w, b):
    return w * x + b


def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def gradients(x, y_true, y_pred):
    n = len(x)
    error = y_pred - y_true
    dw = (2 / n) * np.sum(error * x)
    db = (2 / n) * np.sum(error)
    return dw, db


def train(x, y, lr, steps=200):
    w = 0.0
    b = 0.0
    losses = []

    for step in range(steps):
        y_pred = predict(x, w, b)
        loss = mse_loss(y_pred, y)
        dw, db = gradients(x, y, y_pred)

        w = w - lr * dw
        b = b - lr * db

        losses.append(loss)

        if step % 20 == 0:
            print(
                f"lr={lr} "
                f"step={step} "
                f"loss={loss:.4f} "
                f"w={w:.4f} "
                f"b={b:.4f}"
            )

    return w, b, losses


def main():
    x, y = make_data()

    all_losses = {}

    for lr in [0.0001, 0.01, 1.0]:
        print("=" * 60)
        w, b, losses = train(x, y, lr=lr, steps=200)
        all_losses[lr] = losses
        print(
            f"final lr={lr}: "
            f"w={w:.4f}, "
            f"b={b:.4f}, "
            f"final_loss={losses[-1]:.4f}"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("没有安装 matplotlib，跳过画图。")
        return

    for lr, losses in all_losses.items():
        plt.plot(losses, label=f"lr={lr}")

    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("linear_regression_losses.png", dpi=150)
    print("已保存图片：linear_regression_losses.png")


if __name__ == "__main__":
    main()
```

关键公式：

模型：

$$
\hat{y}=wx+b
$$

均方误差：

$$
L=\frac{1}{n}\sum_i(\hat{y}_i-y_i)^2
$$

梯度：

$$
\frac{\partial L}{\partial w}
=
\frac{2}{n}\sum_i(\hat{y}_i-y_i)x_i
$$

$$
\frac{\partial L}{\partial b}
=
\frac{2}{n}\sum_i(\hat{y}_i-y_i)
$$

结论参考：

- `lr=0.0001`：通常很稳定，但下降很慢。
- `lr=0.01`：通常能较稳定地收敛，`w` 接近 `3`，`b` 接近 `2`。
- `lr=1.0`：容易震荡或发散，loss 可能变得非常大。
- 最终学到的参数不会刚好等于 `3` 和 `2`，因为数据里加入了噪声。

---

## notes.md 参考写法

```markdown
# 模块 1 综合练习记录

## Exercise A

### 我做了什么

读取 JSONL，统计坏行、空文本、重复文本、长度分桶和高频字符。

### 我如何验证

使用 6 行 demo 数据，其中包含正常行、空文本、重复文本、非法 JSON 行和 instruction/output 样本。

### 结果

- 总行数：
- 合法 JSON 记录数：
- 坏行数：
- 空文本数：
- 重复文本数：
- 高频字符：

### 结论

真实数据不能默认每一行都是干净的；读取时必须处理坏行、空字段和重复样本。

## Exercise B

### 我做了什么

练习 NumPy 的 `shape`、`mean`、`reshape`、矩阵乘法和 stable softmax。

### 结论

深度学习代码里最重要的是先写清楚张量形状，再写计算。

## Exercise C

### 我做了什么

用 NumPy 从零实现一元线性回归，比较三个学习率。

### 结论

学习率太小训练慢，合适时稳定下降，太大时可能发散。
```
