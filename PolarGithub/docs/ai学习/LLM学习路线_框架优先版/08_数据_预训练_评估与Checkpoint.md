# 第三遍之一：数据、预训练、评估与 Checkpoint

本章从“程序能够训练”走向“实验结果值得相信”。重点包括数据来源、切分防泄漏、训练预算、验证指标、故障诊断和可恢复 Checkpoint。

## 1. 预训练的任务与边界

自回归预训练反复执行：

$$
\max_\theta
\sum_t\log p_\theta(x_t\mid x_{<t})
$$

等价于最小化平均交叉熵。模型没有人工提供的“语法规则表”，而是在大量下一个 token 预测中学习统计规律与可复用表示。

教学模型与真实预训练的区别主要在规模、数据治理和基础设施，不在最基本的预测目标。

## 2. 数据来源与使用许可

采集任何训练文本前记录：

```text
来源名称与 URL
获取日期
许可证或授权范围
是否允许训练、修改、再分发和商业使用
是否含个人信息、密钥或内部机密
清洗与过滤版本
```

“互联网上可以访问”不等于“可以任意训练和发布”。个人学习也应培养来源与许可意识，公开发布模型、数据或生成内容时尤其重要。

## 3. 数据清洗保留必要结构

常见清洗步骤：

- 统一 UTF-8 与 Unicode 规范形式。
- 去除无法解释的控制字符和损坏编码。
- 按目标语言、长度和质量过滤。
- 检测重复文档和高度相似片段。
- 扫描个人信息、访问令牌、密码与私钥。
- 保留文档边界和来源元数据。

过度清洗也会造成问题：代码缩进、换行、Markdown 结构和中文标点都可能携带语义。清洗规则必须通过抽样对比验证。

## 4. 训练验证测试切分防止泄漏

正确顺序：

```text
原始文档
→ 以文档或来源组为单位去重
→ 划分 train / validation / test
→ 各集合内部独立 tokenize 与切块
```

高风险顺序：

```text
先把文档切成大量重叠窗口
→ 再随机分配窗口
```

相邻窗口可能高度重复并进入不同集合，导致验证损失虚低。

测试集应在开发期间尽量少看，最终用于一次较客观的模型选择评估。

## 5. Packing 与序列边界

固定长度训练中，短样本全部 padding 会浪费计算。Packing 把多个片段拼入长度 $T$ 的训练块。

常见方案：

```text
连续文本流：直接拼接，适合连续语料
EOS 分隔：文档之间加入结束 token
分段 Mask：进一步阻止不同样本之间互相注意
```

是否允许跨文档预测属于数据语义选择，必须在实验记录中说明。

配套代码使用最简单的连续 token 流随机截取，目的是让输入目标错位关系透明，不是生产级数据管道。

## 6. 训练预算的基本单位

只记录 Epoch 在流式或 Packing 训练中不够稳定，更通用的量包括：

```text
step：优化器更新次数
tokens per step：B×T×梯度累积步数
tokens seen：累计处理的训练 token
wall-clock time：实际训练时间
FLOPs 或吞吐：更深入的算力指标
```

近似：

$$
\text{tokens seen}
=
\text{steps}\times B\times T\times A
$$

$A$ 是梯度累积步数。

比较实验时至少保持数据版本、token 数预算、随机种子与评估方式一致。

## 7. 参数与训练内存的组成

显存或内存不只是模型权重：

```text
参数
+ 梯度
+ 优化器状态
+ 激活
+ 临时张量
+ 框架与内核开销
```

AdamW 通常为每个参数维护一阶与二阶状态。训练还要保存反向传播所需激活。Attention 的显式分数矩阵含 $T\times T$，上下文长度增长会显著增加内存与计算。

OOM 调整顺序通常为：

1. 减小 Batch size。
2. 减小 sequence length。
3. 使用梯度累积保持有效 Batch。
4. 使用混合精度与激活检查点。
5. 减小层数或隐藏宽度。
6. 再考虑更复杂的并行和高效内核。

## 8. 训练日志与核心指标

至少记录：

```text
step 与 tokens seen
train loss 与 validation loss
learning rate
gradient norm
tokens per second
设备与峰值内存
定期固定提示生成样例
```

困惑度定义为：

$$
\operatorname{PPL}=e^L
$$

其中 $L$ 是按 token 平均的自然对数交叉熵。困惑度只能在 Tokenizer、数据与损失口径相同时公平比较。

## 9. 评估的三层结构

**实现正确性**

- 张量形状断言。
- Causal Mask 未来权重为零。
- 单批过拟合。
- 保存加载后相同输入 logits 一致。
- 学习率为零时参数不变。

**统计性能**

- 训练、验证、测试 loss 与 perplexity。
- 不同上下文长度或数据子集表现。
- 多随机种子结果的均值与波动。

**行为质量**

- 固定提示生成样例。
- 重复、乱码、事实错误和不安全输出分析。
- 训练数据复述与隐私泄漏检查。

生成样例有直觉价值，但不能代替定量指标。

## 10. 过拟合与欠拟合的曲线判断

| 现象 | 初步判断 | 下一步 |
|---|---|---|
| train 与 val 都高且下降慢 | 欠拟合或优化不足 | 检查实现、训练更久、调学习率或增容量 |
| train 低而 val 明显上升 | 过拟合 | 增数据、正则化、早停或减容量 |
| 两者剧烈震荡 | 更新不稳定 | 降学习率、检查梯度与数据异常 |
| 突然出现 NaN | 数值或数据故障 | 检查首个异常 step、梯度、输入与精度 |
| val 异常好 | 可能数据泄漏 | 检查去重和切分顺序 |

诊断顺序应从数据与实现开始，不应直接用更大模型掩盖问题。

## 11. 可恢复 Checkpoint 的完整内容

推荐保存：

```text
model_state
optimizer_state
scheduler_state（若有）
gradient_scaler_state（混合精度时）
step / epoch / tokens_seen
best_metric
model_config
training_config
tokenizer 文件或版本
data_version
Python、NumPy、PyTorch 与 CUDA 随机状态
代码版本或 Git commit
```

Checkpoint 解决的是“继续同一个实验”，模型权重文件解决的是“加载模型进行推理”。二者范围不同。

## 12. 保存加载的一致性测试

在 `eval()` 和 `no_grad()` 下进行：

```python
model.eval()
with torch.no_grad():
    logits_before, _ = model(test_input)

torch.save(model.state_dict(), path)

reloaded = MiniGPT(config).to(device)
reloaded.load_state_dict(torch.load(path, map_location=device))
reloaded.eval()
with torch.no_grad():
    logits_after, _ = reloaded(test_input)

torch.testing.assert_close(logits_before, logits_after)
```

若忘记 `eval()`，Dropout 会让两次结果不同，不能据此判断保存失败。

## 13. 实验记录的最小规范

每次实验记录：

```text
实验目的
唯一主要变量
数据与 Tokenizer 版本
模型配置
优化器、学习率与训练 token 数
随机种子和设备
训练、验证结果
固定提示生成样例
异常和解释
下一步结论
```

可直接复制本目录 [学习记录模板.md](学习记录模板.md)。

## 14. 从教学代码到正式训练的升级顺序

1. 把语料处理与模型代码分开。
2. 按文档划分数据并保存 Tokenizer。
3. 增加完整验证集遍历和测试集。
4. 增加学习率 warmup 与衰减。
5. 增加自动混合精度、梯度累积和吞吐统计。
6. 增加可恢复的数据迭代位置和完整随机状态。
7. 增加配置文件、日志目录和代码版本记录。
8. 在单 GPU 正确后再进入多 GPU 或分布式训练。

每次升级都保留一个最小正确性测试。

## 15. 本章项目

基于完整程序完成一次受控对照：

```text
固定：语料、随机种子、block_size、batch_size、训练步数
变量：n_layer = 1 与 n_layer = 3
记录：参数量、训练时间、train loss、validation loss、固定提示生成
```

结论必须区分观察与解释。例如：“三层模型验证损失更低”是观察；“更大容量可能更适合当前数据”是解释，不能把一次小实验写成普遍规律。

## 16. 本章验收

- 能说明文档级切分优于切块后随机切分。
- 能区分 step、token 数与 Epoch。
- 能列出训练内存的主要组成。
- 能说明 perplexity 的比较前提。
- 能区分实现正确性、统计性能和行为质量评估。
- 能列出完整恢复训练所需状态。
- 能完成保存加载一致性测试。
- 能提交一个只改变单一变量的实验记录。
