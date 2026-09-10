# 模块 6：SFT、LoRA 与 QLoRA

## 目标

理解微调到底改变什么；准备高质量指令数据；完成一个小型“数学老师”模型的 LoRA/QLoRA 微调，并用固定评估集证明变化，而不是只展示一个好看的回答。

> SFT mask、低秩参数量、缩放、合并与 4-bit 计算路径的逐步推导见 [LoRA、QLoRA 与监督微调推导](06A_LoRA与QLoRA推导.md)。

## 学习导航

先判断问题是否真的需要微调，再学习数据、训练与评估；遇到矩阵和显存公式时对照补充讲义。完成验收与复盘后，最后再做文末 Project。

## 1. 先区分训练方式

- 预训练：大量无标注文本上学习下一个 token，获得通用语言能力。
- 继续预训练：在领域原始文本上继续同一目标，补领域分布。
- SFT：在“指令→理想回答”数据上监督微调，学习行为和格式。
- 偏好优化：利用偏好对让模型更偏向某类回答，例如 DPO；不是本模块必做。
- RAG：推理时提供外部资料，不修改模型参数。

需要新知识且资料常更新，优先 RAG；需要稳定语气、格式、流程，SFT 更合适；两者可以组合。

## 2. SFT 的标签构造

对话序列经过 chat template 变成 token。常见做法只在 assistant 回答 token 上计算 loss，system/user 部分标签设为 ignore index（如 `-100`）。否则模型也被训练去复述用户输入。

```text
<system>你是数学老师</system>     label: ignore
<user>解释勾股定理</user>         label: ignore
<assistant>在直角三角形中……       label: 学习这些 token
```

不同模型的模板不同，必须使用对应 tokenizer 的模板并检查渲染后的文本/token。

## 3. 数据质量

数学老师数据应包含：题目、推理/讲解、最终答案、难度、主题以及可自动验证的信息。不要混入错误答案；高质量的几千条常胜过大量噪声。

切分时按题目模板/来源去重，防止同一道题改几个数字同时出现在训练和测试。测试集应覆盖：

- 直接计算；
- 文字题；
- 要求分步讲解；
- 信息不足应追问；
- 用户给出错误前提；
- 超出能力时诚实说明。

## 4. Full Fine-tuning 与 LoRA

全量微调更新所有权重，质量上限高但显存、存储和训练成本大。

LoRA 冻结原权重 `W`，只学习低秩增量：

$
W'=W+\Delta W,\quad \Delta W=\frac{\alpha}{r}BA
$

若 `W:[d_out,d_in]`，可令 `A:[r,d_in]`、`B:[d_out,r]`，且 `r` 远小于输入输出维度。可训练参数从 `d_out*d_in` 降为 `r*(d_in+d_out)`。

`r` 是容量；`alpha` 控制缩放；LoRA dropout 是正则。常注入 attention 投影和/或 MLP 线性层。目标模块名依模型实现而异，不能盲抄别人的 `q_proj` 列表。

LoRA 省的是可训练参数、梯度和优化器状态；基础模型权重仍要加载。

## 5. QLoRA

QLoRA 通常把冻结的基础模型以 4bit 形式加载，再用较高精度训练 LoRA adapter。关键点：

- 量化的是冻结基础权重，不是简单地“所有计算都 4bit”。
- LoRA 参数和关键计算通常保持 BF16/FP16/FP32。
- NF4 是针对近似正态分布权重设计的 4bit 数据类型思想。
- double quantization 进一步量化量化常数，减少平均存储。
- paged optimizer 等技术用于缓解显存峰值（具体框架实现会变化）。

8bit/4bit 是表示位宽，不自动保证效果、速度或硬件兼容。CPU、不同 GPU 与算子支持差异很大。

## 6. 显存粗算与省显存手段

先列账：基础权重、LoRA 参数、梯度、优化器状态、激活、临时 buffer。可调整：

- batch size 和 gradient accumulation；
- sequence length（对激活影响很大）；
- gradient checkpointing（以重算换显存）；
- 混合精度；
- LoRA rank/目标层；
- 4bit 基座；
- 优化器和 attention 实现。

gradient accumulation 让多个 micro-batch 的梯度累积后更新，近似更大 batch；注意按累积步数缩放 loss，并只在更新时 scheduler.step。

## 7. 训练配置要记录什么

```yaml
base_model: 精确名称与 revision
tokenizer: 精确名称与 revision
dataset_version: 哈希或版本
chat_template: 名称/内容
max_length: 1024
trainable_modules: [q_proj, v_proj]
lora_rank: 16
lora_alpha: 32
learning_rate: 0.0002
epochs: 3
effective_batch_size: 32
precision: bf16
seed: 42
```

还应记录依赖版本、硬件、训练时长、峰值显存和许可证。

## 8. 用 LLaMA-Factory/Unsloth 的正确方式

工具负责工程实现，不替你决定数据是否正确。推荐顺序：

1. 先用 20 条样本跑通并过拟合。
2. 打印模板渲染结果和标签 mask。
3. 用小训练集跑短实验，确认 loss 与生成变化。
4. 再启用完整数据和省显存优化。
5. 导出 adapter 与配置；必要时合并，但保留原 adapter。

项目命令和参数会更新，以当前官方文档为准。不要把博客里的旧命令固化为概念。

## 9. 评估设计

至少比较：base model、fine-tuned model、可选的 base+RAG。指标分层：

- 正确性：数学题可用程序解析最终答案。
- 格式遵循：是否按步骤/JSON/指定语言回答。
- 教学质量：盲评清晰度、是否跳步、是否指出错误。
- 保持能力：通用小集合上是否灾难性遗忘。
- 安全性：是否泄露训练样本、是否服从恶意数据指令。

生成参数必须相同。评估题不能来自训练数据。人工评分需先写 rubric，再隐藏模型身份。

## 10. 常见失败

- 模板不匹配，模型输出控制 token 或胡乱续写。
- 所有 token 都参与 loss，模型学会复述用户。
- padding 未 mask，模型大量学习 `<pad>`。
- 学习率过大导致原能力明显退化。
- 训练集和测试集近重复。
- 只看训练 loss，不做行为评估。
- adapter 加载到错误的 base revision。
- 把“模型口吻更像老师”误认为“数学正确率提高”。

## 验收与复盘

- 能解释 LoRA 参数量为何下降，QLoRA 哪部分被量化。
- 能检查一条样本 token、模板和 loss mask。
- 有不可篡改的测试集与 base 对照。
- 能说明什么时候该用 RAG 而不是微调。
- 训练产物可加载，模型卡和数据卡完整。

## 课后 Project

### Project 8/9：数学老师模型

交付物：

1. 数据卡：来源、许可、清洗、去重、分布和已知偏差。
2. 100～300 道固定测试题，尽可能可自动评分。
3. LoRA 与 QLoRA 各一个小实验，记录显存、速度、效果。
4. Base/LoRA/QLoRA 对照表和失败案例。
5. 推理脚本，确保加载正确 base、adapter 和 tokenizer。
6. 模型卡：用途、限制、不适用场景。
