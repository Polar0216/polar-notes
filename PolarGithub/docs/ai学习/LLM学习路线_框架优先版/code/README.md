# MiniGPT 完整代码运行指南

本目录中的 `mini_gpt_complete.py` 是整套框架优先教程的可运行主线。它不是伪代码，包含字符级 Tokenizer、数据切分、Batch 采样、Embedding、多头因果自注意力、MLP、Transformer Block、交叉熵、AdamW、验证、Checkpoint、恢复训练和自回归生成。

## 1. Conda 环境准备

先打开 Anaconda Prompt 或已经初始化 Conda 的 PowerShell：

```powershell
conda create -n llm-map python=3.11 -y
conda activate llm-map
```

`conda create` 只创建包含 Python 和基础工具的干净环境，不会自动安装 PyTorch。先确认当前命令确实使用 `llm-map` 中的 Python：

```powershell
where.exe python
python -m pip --version
```

两条输出的路径都应包含：

```text
C:\Users\34201\miniconda3\envs\llm-map\
```

此时运行 `python -m pip install ...`，包会安装到 `llm-map`，不会装进 `base` 或系统 Python。

PyTorch 的 CUDA 安装命令会随正式版本变化。打开 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)，依次选择 Windows、Pip、Python 和适合当前驱动的 CUDA 版本，再复制页面生成的命令。支持 CUDA 12.8 时，命令形式通常为：

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

只需学习模型结构或暂时不使用 NVIDIA GPU 时，可在官方选择器中选择 CPU。更完整说明见项目根目录的 [PyTorch 使用指南](../../PyTorch使用指南.md)。

安装完成后检查：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

`torch.cuda.is_available()` 返回 `False` 并不妨碍运行教学模型，它会自动使用 CPU。

如果出现：

```text
ModuleNotFoundError: No module named 'torch'
```

这表示当前 Python 环境中尚未安装 PyTorch，并不表示 Conda 环境创建失败或 MiniGPT 代码损坏。重新核对 `(llm-map)`、`where.exe python` 和 `python -m pip --version`，然后在该环境内完成安装。

## 2. 第一次快速运行

进入本目录：

```powershell
cd "C:\Users\34201\Desktop\ai学习\LLM学习路线_框架优先版\code"
```

先只运行 20 步并打印全局形状：

```powershell
python mini_gpt_complete.py --max-steps 20 --eval-interval 10 --eval-iters 2 --trace
```

应当看见类似的数据流：

```text
token ids             (2, 64)
token embedding       (2, 64, 96)
position embedding    (64, 96)
combined hidden state (2, 64, 96)
transformer block 1   (2, 64, 96)
transformer block 2   (2, 64, 96)
transformer block 3   (2, 64, 96)
final hidden state    (2, 64, 96)
logits                (2, 64, V)
targets               (2, 64)
loss                   scalar
```

这里的 `V` 会由语料中不同字符的数量决定。

## 3. 完整教学训练

CPU 可以先使用较小配置：

```powershell
python mini_gpt_complete.py `
  --device cpu `
  --n-layer 2 `
  --n-head 4 `
  --n-embd 64 `
  --block-size 48 `
  --batch-size 8 `
  --max-steps 500 `
  --eval-interval 50 `
  --prompt "我爱"
```

若 CUDA 可用，可直接运行默认配置：

```powershell
python mini_gpt_complete.py --device cuda --trace
```

训练完成后，验证损失最优的状态会保存到：

```text
mini_gpt_checkpoint.pt
```

## 4. 恢复训练

将总步数提高到 1200，并从已保存状态继续：

```powershell
python mini_gpt_complete.py `
  --resume mini_gpt_checkpoint.pt `
  --output mini_gpt_checkpoint.pt `
  --max-steps 1200
```

恢复时模型结构和 Tokenizer 使用 Checkpoint 中保存的配置。当前语料应与原训练语料保持一致。

## 5. 生成参数实验

较稳定的生成：

```powershell
python mini_gpt_complete.py --max-steps 500 --temperature 0.6 --top-k 10
```

更多样但更容易混乱的生成：

```powershell
python mini_gpt_complete.py --max-steps 500 --temperature 1.2 --top-k 40
```

贪心生成使用温度零：

```powershell
python mini_gpt_complete.py --max-steps 500 --temperature 0
```

每次重新训练会花费时间。如果只是比较采样，后续可在阅读第 9 章后把程序拆成独立的 `train.py` 和 `generate.py`。

## 6. 阅读代码的推荐顺序

不要从文件第一行逐字读到最后一行。按数据流阅读：

1. `main()`：观察训练程序如何连接所有组件。
2. `MiniGPT.forward()`：观察 `[B,T] → [B,T,C] → [B,T,V]`。
3. `TransformerBlock.forward()`：观察 Attention、MLP、LayerNorm 与残差。
4. `CausalSelfAttention.forward()`：观察 `[B,T,C] → [B,H,T,D]`。
5. `get_batch()`：观察输入和目标如何错开一位。
6. 训练循环：观察 `zero_grad → forward → backward → step`。
7. `MiniGPT.generate()`：观察 token 如何接回输入。
8. Checkpoint 函数：观察哪些状态必须保存。

## 7. 常见报错定位

| 现象 | 常见原因 | 处理方式 |
|---|---|---|
| `No module named torch` | 当前 Conda 环境没有安装 PyTorch | 激活 `llm-map` 后重新安装 |
| `n_embd 必须能被 n_head 整除` | 每个头无法获得相同宽度 | 使用 `64/4`、`96/4` 等组合 |
| `语料过短` | 验证集不足一个序列窗口 | 增加文本或减小 `--block-size` |
| CUDA 不可用 | 驱动、安装版本或硬件不匹配 | 先用 `--device cpu` 完成课程 |
| 损失不下降 | 步数过少、学习率异常或代码被改坏 | 恢复默认参数并做单批过拟合测试 |
| 生成文字混乱 | 语料和模型都很小 | 先观察损失与结构，不以语言质量作为唯一验收 |

## 8. 最小验收

完成以下操作即可进入第二遍学习：

- 使用 `--trace` 运行并解释每个打印形状。
- 找到 `MiniGPT.forward()` 中 LM Head 产生 logits 的代码。
- 找到训练循环中的四步更新顺序。
- 找到 `generate()` 中把新 token 接回输入的代码。
- 修改 `--n-layer` 后观察参数量变化。
- 修改 `--temperature` 后比较生成文本差异。
