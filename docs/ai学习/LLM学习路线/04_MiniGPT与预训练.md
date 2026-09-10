# 模块 4：MiniGPT 与预训练

## 目标

组合前面的组件，训练一个教育用途的小型自回归 GPT；理解数据清洗、切分、packing、训练稳定性、checkpoint、评估和计算预算。

> 序列损失、参数量、AdamW、梯度累积和显存账本的详细推导见 [GPT 训练原理、优化器与显存估算](04A_训练原理与显存估算.md)。

## 学习导航

按“先打通 GPT 数据流 → 用补充讲义核对训练原理 → 在小数据上验证 → 完成验收与复盘 → 最后做文末 Project”的顺序推进。先求完整可复现，再扩大模型和数据。

## 1. GPT 的完整数据流

```text
文本
→ tokenizer 得到 ids [B,T]
→ token embedding + position encoding [B,T,C]
→ N 个 Transformer Block [B,T,C]
→ final LayerNorm
→ LM head [B,T,V]
→ 与右移一位的 targets 计算交叉熵
```

输入与目标：若 token 是 `[春, 天, 来, 了]`，训练对可写为：

```text
input : [春, 天, 来]
target: [天, 来, 了]
```

一次前向便并行训练所有位置；生成时却必须逐 token 自回归。

## 2. 模型骨架

```python
class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError("sequence is longer than block_size")
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(B * T, -1), targets.reshape(B * T)
            )
        return logits, loss
```

输入 embedding 与输出 head 可共享权重（weight tying），减少参数并利用输入输出词语义相关性；实现时确保形状一致。

## 3. 参数量和内存先估算

粗略地，Transformer 层参数主要是 attention 的 4 个 `C×C` 矩阵与 MLP 的约 `8C²`，每层约 `12C²`，再加 embedding `V×C`。这只是数量级估算。

训练显存不仅是权重：

```text
权重 + 梯度 + 优化器状态 + 激活 + 临时张量 + KV/框架开销
```

Adam 常保存两份额外状态；混合精度还可能保留 FP32 主权重。序列长度增加也会显著增加激活，朴素 attention 矩阵随 `T²` 增长。

## 4. 数据：质量比“下载得多”更重要

### 来源与许可

记录来源、许可证、采集日期和允许用途。个人笔记可能含隐私；小说和网页可能受版权约束；Wikipedia 也需遵循相应许可与署名要求。

### 清洗

常见步骤：解码与 Unicode 规范化、去控制字符、语言过滤、质量过滤、去重、隐私/密钥扫描、记录文档边界。不要过度清洗到丢失代码缩进、换行或中文标点。

### 切分避免泄漏

先按文档划分 train/val/test，再从各自文档切块。若先切成 chunk 再随机分，几乎相同的相邻块可能落入不同集合，验证损失虚低。

## 5. Packing

固定长度训练若每条短文本都 padding，会浪费算力。Packing 把多条 token 序列拼入固定长度 block。

两种语义：

- 连续语料：可跨文档预测，简单但边界可能不自然。
- 用 `<eos>` 分隔：模型学到文档结束；必要时使用分段 attention/loss mask 避免跨样本污染。

最简单的连续流：

```python
def get_batch(data, batch_size, block_size, device):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
```

## 6. 初始化、优化与稳定性

入门可用 AdamW。必须理解：

- learning rate 决定步长。
- batch size 影响梯度噪声和内存。
- weight decay 约束权重，通常不施加于 bias 和 LayerNorm 参数。
- warmup 让训练开始时逐步提高学习率。
- gradient clipping 限制异常大梯度。
- mixed precision 加速和省显存，但可能溢出/下溢。

训练日志至少记录 step、tokens seen、train/val loss、学习率、梯度范数、吞吐 tokens/s、显存峰值和样例输出。

### 先过拟合一个 batch

固定同一小 batch 反复训练，模型应能把 loss 降得很低。若做不到，先查实现；若能做到却完整训练不行，再查数据、采样和超参数。

## 7. Checkpoint 的正确含义

可恢复训练的 checkpoint 包含：模型、优化器、scheduler、step/epoch、最佳指标、随机数状态、模型/数据配置、tokenizer 版本。否则“继续训练”可能不是同一实验。

保存策略：

- 定期保存 `last` 用于断点恢复。
- 验证集最佳保存 `best` 用于最终评估。
- 用临时文件写完再原子替换，降低中途损坏风险。
- 定期实际测试加载，不要等崩溃后才发现 checkpoint 无法用。

## 8. 评估

### 定量

在固定验证 token 上计算平均交叉熵/困惑度；对比必须使用同一 tokenizer、同一数据和同一 token 计数方式。

### 定性

用固定 prompts 和固定随机种子定期生成，观察：局部语法、重复、记忆训练文本、主题连贯性、结束行为。挑好看的样例不算评估，应保存完整样例和失败样例。

### 数据记忆检查

小数据训练容易复述语料。对训练片段前缀生成，检查长串精确匹配；敏感数据绝不能靠“模型应该不会记住”来保护。

## 9. 阅读 `minGPT` / `nanoGPT` 的顺序

1. 先看配置和 README，运行最小示例。
2. 找数据预处理，确认产物和 dtype。
3. 看模型类，从 `forward` 沿形状追踪。
4. 看训练循环、优化器分组和评估。
5. 看采样脚本。
6. 最后看分布式、编译和性能优化。

不要第一天钻进分布式训练细节；那会遮住主数据流。

## 验收与复盘

- 模型能在极小 batch 上过拟合。
- 训练和验证 loss 可视化，checkpoint 可真正恢复。
- 参数量估算与程序统计数量级一致。
- 能解释 packing、泄漏、困惑度限制和训练/生成的差异。
- README 让别人从零复现实验，并明确数据许可。

## 课后 Project

### Project 5：MiniGPT

建议三步：

1. 字符级小语料，模型约 0.1M～2M 参数，CPU/GPU 打通。
2. 自己训练 BPE，扩大到数百万参数，比较字符级和 BPE。
3. 阅读 `minGPT`/`nanoGPT`，把自己的模块对应到工业化写法。

不要一开始追求“几十 M 参数”。先让 1M 模型完整可复现，再逐级扩大。

### Project 6：Pretraining 实验

选取合法的小规模中文语料，建立以下对照：

- context 64 vs 256；
- 1 层 vs 4 层；
- 字符 tokenizer vs BPE；
- 有无去重；
- 固定训练 token 数，而不是只比较 epoch。

报告中回答：哪个变量改变了参数量、速度、显存、验证损失和生成质量？结果是否符合假设？
