# 第三遍之二：推理、采样、KV Cache、量化与部署

本章从训练完成的模型出发，解释一次请求如何变成连续 token，以及推理系统如何减少重复计算、降低内存并稳定地提供服务。

## 1. 推理链路在全局地图中的位置

```text
用户文本
→ Tokenizer.encode
→ prompt ids
→ prefill：一次处理已有提示
→ 得到第一个 next-token logits
→ decode：逐 token 生成
→ 停止条件
→ Tokenizer.decode
→ 输出文本
```

推理不会调用 `loss.backward()` 或 `optimizer.step()`，模型参数保持不变。

## 2. 自回归生成的基础循环

最直接实现：

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens):
    model.eval()
    for _ in range(max_new_tokens):
        idx_context = idx[:, -model.config.block_size:]
        logits, _ = model(idx_context)
        next_logits = logits[:, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx
```

停止条件通常包括：

- 生成 EOS token；
- 达到 `max_new_tokens`；
- 达到上下文长度上限；
- 命中应用层停止串；
- 用户取消或服务超时。

只按字符串截断可能误伤 token 边界，生产实现应明确 Tokenizer 与停止规则的关系。

## 3. 解码策略控制概率到 token 的选择

**贪心解码**

$$
x_{t+1}=\arg\max_i z_i
$$

每次选择最高分 token。结果确定，但容易进入重复模式，也不保证整段序列全局最优。

**温度采样**

$$
p_i=\operatorname{softmax}\left(\frac{z_i}{\tau}\right)
$$

- 较低温度让分布集中。
- 较高温度增加低概率候选机会。
- 温度不改变模型知识，只改变选择随机性。

**Top-k**

保留分数最高的 $k$ 个候选，把其余设为 $-\infty$ 后归一化。

**Top-p**

按概率从高到低排序，保留累计概率达到阈值 $p$ 的最小候选集合。候选数量随分布尖锐程度变化。

策略不是越多越好。建立基线时先固定提示、模型、随机种子和最大生成长度，只改变一种采样参数。

## 4. Prefill 与 Decode 的计算差异

推理通常分两阶段：

| 阶段 | 输入 | 主要特征 |
|---|---|---|
| Prefill | 完整 Prompt，长度 $T$ | 多位置并行，计算并保存各层 K、V |
| Decode | 每次一个新 token | 逐步串行，读取历史 KV Cache |

首 token 延迟常受 Prompt 长度和 Prefill 影响；后续 token 速度受 Decode、内存带宽与 Batch 调度影响。

常见服务指标：

```text
TTFT：Time To First Token，首 token 延迟
TPOT：Time Per Output Token，后续每 token 时间
吞吐：单位时间处理的 token 或请求
端到端延迟：请求到完整响应完成
并发能力：目标延迟下可服务的并行请求数
```

## 5. KV Cache 消除历史投影重复计算

朴素生成每加入一个 token，都把整个历史序列重新送进模型。对每一层 Attention，历史 token 的 K、V 在参数不变时不会改变。

KV Cache 保存每一层过去位置的 Key 与 Value：

```text
第 1 步：计算 prompt 的 K、V 并缓存
第 2 步：只计算新 token 的 Q、K、V；Q 读取缓存 K、V
第 3 步：把新 K、V 追加到缓存；继续
```

它减少重复计算，但带来额外内存。粗略元素数：

$$
2\times N\times B\times H_{kv}\times T\times D
$$

其中 2 表示 K 与 V，$N$ 为层数，$H_{kv}$ 为 KV 头数。

KV Cache 不缓存 Query，因为历史 Query 的输出已经使用过；新 token 只需要新的 Query 去读取全部历史 K、V。

## 6. MHA、MQA 与 GQA 的 KV 头数量

| 结构 | Query 头 | KV 头 | 主要取舍 |
|---|---:|---:|---|
| MHA | $H$ | $H$ | 表达灵活，KV Cache 较大 |
| MQA | $H$ | 1 | Cache 小，多个 Q 头共享同一组 K、V |
| GQA | $H$ | 小于 $H$ 且大于 1 | 在质量和缓存之间折中 |

它们不会改变自回归依赖，只减少 KV 参数与缓存规模或提高 Decode 效率。

## 7. 量化降低权重与缓存成本

量化用较低位宽表示数值，例如把 FP16 权重转换为 INT8 或 4-bit 表示。简化的仿射量化：

$$
q=\operatorname{round}\left(\frac{x}{s}\right)+z
$$

反量化近似恢复：

$$
\hat x=s(q-z)
$$

$s$ 是尺度，$z$ 是零点。

需要区分：

- 权重量化：主要减少模型权重存储与带宽。
- 激活量化：进一步优化计算，但校准更困难。
- KV Cache 量化：降低长上下文推理缓存。
- 训练量化与推理量化：目标和数值要求不同。

量化可能导致精度下降，必须用任务指标和固定样例验证，不能只看模型文件大小。

## 8. 推理引擎承担模型之外的系统工作

成熟推理引擎通常还负责：

```text
模型权重加载与设备放置
高效 Attention 内核
KV Cache 内存管理
动态或连续批处理
请求排队与取消
流式 token 输出
多 GPU 并行
量化算子
吞吐、延迟和错误监控
```

教学代码用 Python 循环清楚展示算法，但不适合高并发服务。理解算法和选择工程引擎属于两个层次。

## 9. 最小 API 的边界设计

一个文本生成接口至少定义：

```json
{
  "prompt": "我爱",
  "max_new_tokens": 80,
  "temperature": 0.8,
  "top_k": 20,
  "seed": 42
}
```

响应应包含：

```json
{
  "text": "我爱……",
  "prompt_tokens": 2,
  "generated_tokens": 80,
  "finish_reason": "length",
  "latency_ms": 1234
}
```

服务层必须限制输入长度、输出长度、并发、超时与参数范围，不能把任意用户参数直接传入底层执行。

## 10. 推理安全与可观测性

需要记录但避免泄露敏感内容：

```text
请求 id
模型与 Tokenizer 版本
采样参数
token 数与耗时
错误类型
资源使用
安全策略结果
```

不要默认记录完整用户 Prompt 和输出；日志需要脱敏、访问控制与保留期限。

加载第三方模型时还要检查来源、许可证、文件格式与自定义代码。不要对不可信模型文件启用任意远程代码执行。

## 11. 基准测试的公平条件

比较两个推理方案时固定：

- 相同模型与量化版本；
- 相同硬件和软件环境；
- 相同 Prompt 长度与输出长度分布；
- 相同 Batch 或并发水平；
- 相同预热过程；
- 相同采样与停止条件。

同时报告延迟分位数和吞吐，例如 P50、P95、P99。只报告最快一次没有代表性。

## 12. 从教学生成到本地部署的路线

1. 用 `mini_gpt_complete.py` 理解朴素生成。
2. 手工增加独立加载 Checkpoint 的生成脚本。
3. 在小模型中实现单层 KV Cache 并与朴素结果对照。
4. 使用成熟开源模型与官方推荐推理工具。
5. 建立固定 Prompt 集合与性能测试脚本。
6. 包装受限 API，增加超时、并发、流式输出和监控。
7. 根据硬件与质量要求评估量化。

## 13. 本章验收

- 能画出 Prefill 与 Decode 两阶段。
- 能解释温度、Top-k 与 Top-p 的不同作用。
- 能说明 KV Cache 保存 K、V 而不保存历史 Query 的原因。
- 能写出 KV Cache 大小涉及的主要维度。
- 能区分 MHA、MQA 与 GQA 的 KV 头数量。
- 能说明量化需要质量评估，不能只看内存。
- 能定义 TTFT、TPOT、吞吐和端到端延迟。
- 能为最小生成 API 写出输入限制与停止条件。

更详细的推导可查原版 `../LLM学习路线/05A_采样_KVCache与量化推导.md`。
