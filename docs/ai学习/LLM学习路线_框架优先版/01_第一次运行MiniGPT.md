# 第一遍之一：第一次运行完整 MiniGPT

本章的目标不是从零写出模型，而是先让一台结构完整的小 GPT 在本机运行起来。你会看到数据怎样进入模型、形状怎样变化、损失怎样产生以及文本怎样生成。

## 1. 本章在全局地图中的位置

本章直接运行整条链路：

```text
tiny_corpus.txt
→ CharTokenizer
→ get_batch
→ MiniGPT
→ logits 与 loss
→ backward 与 AdamW
→ checkpoint
→ generate
```

第一次运行时暂不展开 `CausalSelfAttention` 的内部公式，只确认它接收 `[B,T,C]`，并返回相同形状的上下文表示。内部计算将在第五章逐步展开。

## 2. 环境与文件结构

建议先阅读 [code/README.md](code/README.md)，使用 Conda 创建独立环境。完成后目录结构如下：

```text
LLM学习路线_框架优先版/
├── 00_LLM全局地图.md
├── 01_第一次运行MiniGPT.md
└── code/
    ├── README.md
    ├── mini_gpt_complete.py
    └── tiny_corpus.txt
```

三个文件各有不同职责：

| 文件 | 职责 |
|---|---|
| `tiny_corpus.txt` | 模型学习预测下一个字符所用的数据 |
| `mini_gpt_complete.py` | 模型、训练、验证、保存与生成的完整程序 |
| `mini_gpt_checkpoint.pt` | 程序运行后产生的训练状态，不需要手工编辑 |

## 3. 第一次运行与输出阅读

进入代码目录并运行快速检查：

```powershell
cd "C:\Users\34201\Desktop\ai学习\LLM学习路线_框架优先版\code"
conda activate llm-map
python mini_gpt_complete.py --max-steps 20 --eval-interval 10 --eval-iters 2 --trace
```

如果这里出现 `ModuleNotFoundError: No module named 'torch'`，含义只是当前 Conda 环境还没有安装 PyTorch，不是模型代码损坏。按 [code/README.md](code/README.md) 的顺序核对 `(llm-map)`、`where.exe python` 和 pip 路径，再把 PyTorch 安装到这个环境中。

第一次看输出只追踪三件事：

```text
token ids 的 [B,T]
hidden state 的 [B,T,C]
logits 的 [B,T,V]
```

`parameters`、`grad_norm`、Checkpoint 内部字段和 Attention 细节此时都允许暂时不理解。它们不会阻挡第一次确认主数据流。

输出大致分为四组。

**环境与规模信息**

```text
device=cpu
vocab_size=...
train_tokens=...
validation_tokens=...
parameters=...
```

- `device` 表示计算发生在 CPU 还是 CUDA GPU。
- `vocab_size` 是语料中不同字符构成的词表大小 $V$。
- `train_tokens` 与 `validation_tokens` 表示训练、验证各有多少 token。
- `parameters` 是模型中所有可学习数字的数量。

**前向传播形状**

```text
token ids             (2, 64)
token embedding       (2, 64, 96)
position embedding    (64, 96)
combined hidden state (2, 64, 96)
transformer block 1   (2, 64, 96)
final hidden state    (2, 64, 96)
logits                (2, 64, V)
targets               (2, 64)
loss                   scalar
```

这正是全局地图中的主线：

$$
[B,T]=[2,64]
\longrightarrow
[B,T,C]=[2,64,96]
\longrightarrow
[B,T,V]
$$

位置向量显示为 `(64,96)` 而不是 `(2,64,96)`，是因为同一批样本共享位置编号。它与 token 向量相加时，PyTorch 沿 Batch 维自动广播。

**训练日志**

```text
step=    0 train_loss=... validation_loss=...
train step=    0 batch_loss=... grad_norm=... elapsed=...
```

- `batch_loss` 是当前随机 Batch 的损失，会有波动。
- `train_loss` 是多个训练 Batch 的平均估计。
- `validation_loss` 使用未参与参数更新的数据，反映泛化情况。
- `grad_norm` 是本轮梯度整体大小，用于发现梯度异常。

20 步只用于确认程序能运行，不能期待生成流畅文本。

**生成结果**

程序最后从提示“我爱”开始逐字符生成。小语料、小模型和短训练会产生重复或不通顺文本。第一次验收关注的是：程序能否生成、输出是否来自词表、长度是否增加，而不是文学质量。

## 4. 先从 `main()` 阅读整机连接

打开 [code/mini_gpt_complete.py](code/mini_gpt_complete.py)，先搜索：

```python
def main() -> None:
```

不要急着读上方每个类。先在 `main()` 中找到以下顺序：

```text
解析命令行参数
→ 选择设备并固定随机种子
→ 读取文本
→ 创建或恢复 Tokenizer 和模型配置
→ 文本编码并切分训练集、验证集
→ 创建模型与优化器
→ 训练循环
→ 载入最佳 Checkpoint
→ 生成文本
```

这一步建立“程序框架”。以后阅读任何 LLM 训练仓库，都可以先寻找相同角色，而不是从第一个文件逐行读。

## 5. `forward()` 中的模型主线

搜索：

```python
class MiniGPT(nn.Module):
```

再找到 `forward()`。它的核心顺序是：

```python
token_vectors = self.token_embedding(idx)
position_vectors = self.position_embedding(positions)
x = self.dropout(token_vectors + position_vectors)

for block in self.blocks:
    x = block(x)

hidden = self.final_norm(x)
logits = self.lm_head(hidden)
```

把变量翻译成人话：

```text
idx：离散的字符编号
token_vectors：字符本身的初始向量
position_vectors：字符所在位置的向量
x：当前隐藏状态
block(x)：结合上下文并加工特征
hidden：最后归一化后的上下文表示
logits：对词表中每个候选字符的分数
```

`forward()` 没有显式调用 `backward()`。前向传播负责产生结果与损失，反向传播由训练循环在外部触发。

## 6. 训练循环中的四个关键动作

在 `main()` 下半部分找到：

```python
optimizer.zero_grad(set_to_none=True)
_, loss = model(x, y)
loss.backward()
optimizer.step()
```

中间还有梯度裁剪：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
```

完整顺序可理解为：

1. 清除上一次训练留下的梯度。
2. 用当前参数进行预测并计算损失。
3. 从损失反向计算每个参数的梯度。
4. 限制异常大的整体梯度。
5. 让 AdamW 根据梯度更新参数。

训练循环不断重复，模型对语料中的下一个字符预测逐渐改善。

## 7. 生成循环中的自回归动作

找到 `MiniGPT.generate()`：

```python
for _ in range(max_new_tokens):
    idx_context = idx[:, -self.config.block_size:]
    logits, _ = self(idx_context)
    next_logits = logits[:, -1, :]
    probabilities = F.softmax(next_logits, dim=-1)
    next_id = torch.multinomial(probabilities, num_samples=1)
    idx = torch.cat((idx, next_id), dim=1)
```

这里只取 `logits[:, -1, :]`，因为当前只需要最后位置对“下一个 token”的预测。采样得到的新编号通过 `torch.cat` 接回原序列，然后下一轮重新计算。

生成阶段使用 `@torch.no_grad()`，因为不需要计算梯度或更新模型。

## 8. 三个低风险修改实验

第一次学习只修改命令行参数，不改模型代码。

**实验一：改变层数**

```powershell
python mini_gpt_complete.py --n-layer 1 --max-steps 20 --eval-iters 2
python mini_gpt_complete.py --n-layer 4 --max-steps 20 --eval-iters 2
```

记录参数量和运行速度。层数增加时参数量与计算量都会增加。

**实验二：改变序列长度**

```powershell
python mini_gpt_complete.py --block-size 16 --max-steps 20 --eval-iters 2 --trace
python mini_gpt_complete.py --block-size 64 --max-steps 20 --eval-iters 2 --trace
```

观察 $T$ 如何影响所有包含序列维的张量。

**实验三：改变采样温度**

完成较长训练后，比较：

```text
temperature = 0     贪心选择，稳定但容易重复
temperature = 0.6   分布更集中
temperature = 1.2   分布更平坦，更多样也更容易出错
```

## 9. 本章验收

满足以下条件即可继续，不要求解释 Attention 公式：

- 能运行 20 步快速检查并看到生成结果。
- 能指出训练数据、程序、Checkpoint 分别在哪个文件。
- 能根据 trace 说出 `[B,T] → [B,T,C] → [B,T,V]`。
- 能在代码中找到 `MiniGPT.forward()`、训练四步和 `generate()`。
- 能解释 `loss.backward()` 与 `optimizer.step()` 的职责不同。
- 能说明小模型生成混乱不等于数据流没有跑通。

下一章将把一次训练和一次生成并排展开：[02_训练与生成的完整闭环.md](02_训练与生成的完整闭环.md)。
