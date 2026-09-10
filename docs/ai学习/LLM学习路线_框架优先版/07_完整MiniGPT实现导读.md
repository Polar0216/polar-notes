# 第二遍之五：完整 MiniGPT 的装配与源码导读

本章以 `code/mini_gpt_complete.py` 为唯一主线，把已经拆开的组件重新装成可训练、可验证、可保存、可恢复和可生成的完整系统。建议一边阅读一边在编辑器中搜索类名与函数名。

阅读源码时先确认每个部分解决的实际问题，再看 Python 写法。下面这张表是本章的阅读索引：

| 部分 | 接收什么 | 产生什么 | 解决的问题 |
|---|---|---|---|
| `CharTokenizer` | 字符串 | token id | 把文本转换成模型可计算的整数 |
| Token 与位置 Embedding | token id、位置编号 | `[B,T,C]` | 为每个位置建立初始向量表示 |
| `TransformerBlock` | `[B,T,C]` | `[B,T,C]` | 交换上下文信息并加工每个位置的表示 |
| `lm_head` | `[B,T,C]` | `[B,T,V]` | 给每个位置的全部候选 token 打分 |
| 交叉熵损失 | logits、targets | 标量 loss | 衡量下一个 token 的预测误差 |
| `backward` 与优化器 | loss、模型参数 | 更新后的参数 | 计算影响方向并执行参数更新 |
| `generate` | 已有 token 序列 | 延长后的序列 | 反复预测并追加下一个 token |

这七部分合在一起才是一个可运行的语言模型。某一段代码看不懂时，先确认它位于哪一行，再追踪输入和输出形状。

## 1. 程序的分层结构

源码可以分成六层：

```text
配置层
  GPTConfig

文本接口层
  CharTokenizer

模型组件层
  CausalSelfAttention
  FeedForward
  TransformerBlock
  MiniGPT

数据层
  split_token_stream
  get_batch

训练基础设施层
  estimate_loss
  build_optimizer
  save_checkpoint / load_checkpoint

入口与编排层
  parse_arguments
  main
```

阅读大型仓库时也应先画出类似层次，避免在工具函数中迷失。

## 2. `GPTConfig` 固定结构契约

配置对象包含：

```python
@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 96
    dropout: float = 0.1
```

参数关系：

| 配置 | 决定的结构 |
|---|---|
| `vocab_size` | Token Embedding 行数与 LM Head 输出宽度 |
| `block_size` | 最大上下文长度与因果 Mask 大小 |
| `n_layer` | Transformer Block 重复次数 |
| `n_head` | Attention 并行头数 |
| `n_embd` | 隐藏状态宽度 $C$ |
| `dropout` | 训练时随机失活比例 |

约束：

$$
n_{\text{embd}}\bmod n_{\text{head}}=0
$$

结构配置必须随 Checkpoint 保存，否则无法准确重建模型形状。

## 3. `CharTokenizer` 建立可恢复词表

Tokenizer 从语料收集不同字符，创建双向映射：

```text
stoi: token string → integer id
itos: integer id → token string
```

`state_dict()` 保存有序 token 列表，而不是只保存语料。恢复时必须保持每个 id 对应同一个字符：

```python
{
    "tokens": ["<unk>", "\n", "。", "我", "爱", ...]
}
```

字符排序只用于第一次创建稳定顺序。恢复已有模型时，应读取 Checkpoint 中的列表，不能根据新语料重新排序。

## 4. `CausalSelfAttention` 完成位置间通信

初始化阶段创建：

```python
self.qkv = nn.Linear(C, 3 * C)
self.output_projection = nn.Linear(C, C)
self.causal_mask = lower_triangular_matrix
```

前向阶段的数据流：

```text
x [B,T,C]
→ qkv projection [B,T,3C]
→ split q,k,v: each [B,T,C]
→ split heads: each [B,H,T,D]
→ q @ kᵀ [B,H,T,T]
→ scale + causal mask + softmax
→ weights @ v [B,H,T,D]
→ merge heads [B,T,C]
→ output projection [B,T,C]
```

这一组件最常见的错误位置：

- `transpose` 维度写错；
- `view` 前张量不连续；
- Mask 切片没有匹配当前 $T$；
- Softmax 沿错维度；
- 合并头时没有恢复到 `[B,T,C]`。

配套代码使用 `.transpose(...).contiguous().view(...)` 明确恢复连续存储后再重塑。

## 5. `FeedForward` 完成逐位置加工

结构：

```text
[B,T,C]
→ Linear(C,4C)
→ GELU
→ Linear(4C,C)
→ Dropout
→ [B,T,C]
```

虽然输入有 $B$ 和 $T$ 两个前导维，Linear 只改变最后一维。所有位置共享相同 MLP 参数。

`4C` 是常见教育实现选择，并非所有现代 LLM 都严格使用这个倍率或 GELU；一些模型使用 SwiGLU 等变体。先掌握共同骨架，再研究变体。

## 6. `TransformerBlock` 组合两类子层

```python
x = x + self.attention(self.norm_before_attention(x))
x = x + self.mlp(self.norm_before_mlp(x))
```

拆开第一行：

```text
x_original = x
x_normalized = LayerNorm(x_original)
attention_update = Attention(x_normalized)
x = x_original + attention_update
```

第二行做相同结构，但更新由 MLP 产生。

一个 Block 不改变 $B$、$T$、$C$，因此多个 Block 可以用 `ModuleList` 顺序堆叠。

## 7. `MiniGPT.__init__()` 组装整台模型

创建顺序：

```python
self.token_embedding = nn.Embedding(V, C)
self.position_embedding = nn.Embedding(T_max, C)
self.blocks = nn.ModuleList([...])
self.final_norm = nn.LayerNorm(C)
self.lm_head = nn.Linear(C, V, bias=False)
```

`ModuleList` 与普通 Python list 的关键区别：PyTorch 能发现其中子模块的参数，使它们出现在 `model.parameters()`、`state_dict()` 和设备迁移中。

最后执行权重共享：

```python
self.lm_head.weight = self.token_embedding.weight
```

Embedding 权重形状是 `[V,C]`；PyTorch Linear 权重也保存为 `[V,C]`，所以可以指向同一参数。这样输入 token 表示与输出词表分类共享权重，减少约 $V\times C$ 个独立参数。

## 8. `MiniGPT.forward()` 连接预测与损失

输入检查：

```python
B, T = idx.shape
if T > block_size:
    raise ValueError(...)
```

嵌入阶段：

```python
positions = torch.arange(T, device=idx.device)
token_vectors = self.token_embedding(idx)        # [B,T,C]
position_vectors = self.position_embedding(positions)  # [T,C]
x = token_vectors + position_vectors             # [B,T,C]
```

主干与输出：

```python
for block in self.blocks:
    x = block(x)

hidden = self.final_norm(x)
logits = self.lm_head(hidden)                     # [B,T,V]
```

有 targets 时计算 loss，没有 targets 时只返回 logits：

```python
loss = None
if targets is not None:
    loss = F.cross_entropy(
        logits.reshape(B * T, -1),
        targets.reshape(B * T),
    )
return logits, loss
```

这种接口让同一个模型既能训练，也能生成。

## 9. `get_batch()` 构造自回归监督

函数先随机选择 $B$ 个起点：

```python
starts = torch.randint(0, len(data) - block_size, (batch_size,))
```

对每个起点构造：

```python
x = data[i : i + block_size]
y = data[i + 1 : i + block_size + 1]
```

最后 `torch.stack`：

```text
B 个 [T] → [B,T]
```

再调用 `.to(device)`，保证数据与模型在同一设备。CPU 张量不能直接与 CUDA 参数运算。

## 10. `estimate_loss()` 保持验证纯净

验证函数使用：

```python
@torch.no_grad()
def estimate_loss(...):
    model.eval()
    ...
    model.train()
```

它不会调用：

```python
loss.backward()
optimizer.step()
```

因此验证数据只衡量模型，不修改模型。函数在多个随机 Batch 上求平均，减少单一 Batch 带来的偶然波动。

真实项目最好遍历完整验证集；教学代码为了速度使用抽样估计。

## 11. `build_optimizer()` 区分权重衰减参数

配套实现按参数维度分组：

```python
if parameter.dim() >= 2:
    decay_parameters.append(parameter)
else:
    no_decay_parameters.append(parameter)
```

矩阵权重使用 weight decay；bias 与 LayerNorm 的一维参数不衰减。这是常见简化策略。

随后创建：

```python
torch.optim.AdamW(parameter_groups, lr=learning_rate)
```

参数分组属于优化策略，不改变模型前向结构。

## 12. `main()` 中的完整训练循环

训练循环的主干：

```python
for step in range(start_step, max_steps):
    if should_evaluate:
        losses = estimate_loss(...)
        if validation_improved:
            save_checkpoint(...)

    x, y = get_batch(...)
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    clip_grad_norm_(...)
    optimizer.step()
```

控制流可以拆成两种节奏：

```text
每一步：取 Batch → 前向 → 反向 → 更新
每隔若干步：训练评估 → 验证评估 → 必要时保存
```

日志节奏、评估节奏与保存节奏可以不同，不应把它们误认为模型结构。

## 13. Checkpoint 的保存与恢复

保存内容：

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "config": asdict(model.config),
    "tokenizer": tokenizer.state_dict(),
    "step": step,
    "best_validation_loss": best_validation_loss,
    "torch_rng_state": torch.get_rng_state(),
}
```

恢复顺序：

```text
读取 Checkpoint
→ 恢复 Tokenizer 与 GPTConfig
→ 按配置创建模型和优化器
→ load_state_dict
→ 恢复 step、最佳指标与随机状态
→ 继续训练
```

先创建结构再加载数字，因为 `state_dict` 本身主要保存参数张量，不负责定义 Python 类结构。

## 14. `generate()` 的推理闭环

每一轮：

```text
裁剪到最后 block_size 个 token
→ 模型前向
→ 取最后位置 logits [B,V]
→ 温度缩放
→ 可选 Top-k 过滤
→ Softmax
→ 抽样一个 id [B,1]
→ 拼到原序列右侧
```

如果 `temperature <= 0`，程序使用 `argmax` 进行贪心选择。否则使用 `torch.multinomial` 按概率抽样。

`@torch.no_grad()` 与 `model.eval()` 共同保证生成不构建训练计算图，并关闭 Dropout 随机行为。

## 15. 参数量的结构估算

主要参数数量级：

```text
Token Embedding：V×C
Position Embedding：T_max×C
每层 Attention：约 4C²
每层 MLP：约 8C²
每层 LayerNorm：约 4C
Final LayerNorm：约 2C
LM Head：与 Token Embedding 共享
```

每个 Block 主体约为：

$$
12C^2
$$

因此层数 $N$ 增加时参数量近似线性增长，隐藏宽度 $C$ 增加时主体参数近似按平方增长。

## 16. 源码修改的渐进顺序

按风险从低到高修改：

1. 修改命令行超参数并观察参数量、速度和损失。
2. 在已有位置增加 shape 断言和日志。
3. 增加独立的单批过拟合函数。
4. 将训练与生成拆成两个脚本。
5. 增加学习率调度和更完整的日志。
6. 替换位置编码或 MLP 激活函数。
7. 使用高效 Attention 实现并进行数值对照。

每次只做一种改变，并保留能够运行的上一个版本。

## 17. 第二遍综合验收

- 能从 `main()` 追踪到模型、数据、损失、优化器和生成。
- 能为 Attention 中的主要张量标注完整形状。
- 能说明 `ModuleList` 对参数注册的重要性。
- 能说明 `forward()` 为何允许 `targets=None`。
- 能解释输入与目标错开一位的代码。
- 能列出恢复训练所需 Checkpoint 内容。
- 能估算层数与隐藏宽度对参数量的影响。
- 能独立修改一个低风险配置并形成对照记录。

通过后进入第三遍：[08_数据_预训练_评估与Checkpoint.md](08_数据_预训练_评估与Checkpoint.md)。
