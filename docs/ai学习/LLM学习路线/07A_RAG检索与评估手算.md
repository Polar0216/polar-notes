# RAG 检索、排序与评估手算

## 讲义定位与使用方式

这是模块 7 的检索与评估补充。先读主模块的 RAG 流程，再在这里手算排序和指标；完成后用这些指标审视文末 Project 的实验结果。

## 1. 先建立一个玩具知识库

有四个文档块：

- $d_1$：Python 使用缩进表示代码块；
- $d_2$：PyTorch 是深度学习框架；
- $d_3$：Python 虚拟环境可以隔离依赖；
- $d_4$：Git 用于版本控制。

查询：

$$
q=\text{“如何隔离 Python 项目依赖？”}
$$

直觉上的相关文档是 $d_3$。RAG 的检索阶段要在不查看参考答案的前提下，把它排到前面。

## 2. 余弦相似度手算

假设经过一个极简 embedding，查询与两个文档为：

$$
q=
\begin{bmatrix}
1&1
\end{bmatrix}
$$

$$
d_1=
\begin{bmatrix}
1&0
\end{bmatrix},
\qquad
d_3=
\begin{bmatrix}
2&2
\end{bmatrix}
$$

余弦相似度：

$$
\cos(q,d)
=
\frac{q\cdot d}
{\lVert q\rVert_2\lVert d\rVert_2}
$$

对 $d_1$：

$$
\cos(q,d_1)
=
\frac{1}
{\sqrt2\times1}
\approx0.707
$$

对 $d_3$：

$$
\cos(q,d_3)
=
\frac{4}
{\sqrt2\times\sqrt8}
=1
$$

因此 $d_3$ 排在 $d_1$ 前。

注意：这个二维向量只是教学例子。真实 embedding 维度更高，坐标本身通常不可直接解释。

## 3. 点积与余弦为何可能排序不同

点积：

$$
q\cdot d
=
\lVert q\rVert
\lVert d\rVert
\cos\theta
$$

它同时受方向和长度影响。余弦只比较方向。若模型训练目标预期归一化向量，却直接使用未归一化点积，文档向量长度可能意外影响排序。

## 4. BM25 的核心思想

BM25 是稀疏关键词检索方法。简化公式：

$$
\operatorname{BM25}(q,d)
=
\sum_{t\in q}
\operatorname{IDF}(t)
\frac{
f(t,d)(k_1+1)
}{
f(t,d)+
k_1
\left(
1-b+b\frac{|d|}{\operatorname{avgdl}}
\right)
}
$$

其中：

- $f(t,d)$：词 $t$ 在文档 $d$ 中出现次数；
- $|d|$：文档长度；
- $\operatorname{avgdl}$：平均文档长度；
- $\operatorname{IDF}(t)$：稀有词的重要程度；
- $k_1,b$：控制词频饱和和长度归一化。

直觉：

- 一个词出现一次到两次有帮助，但出现一百次不应线性提高一百倍；
- 稀有产品编号比“这个”“如何”等常见词更有区分力；
- 长文档天然包含更多词，需要长度校正。

## 5. 为什么需要 Hybrid Search

向量检索擅长语义近义，例如“隔离依赖”与“虚拟环境”；BM25 擅长精确字符串，例如错误码、零件编号和函数名。

两种原始分数不可直接相加，因为尺度不同。常用 Reciprocal Rank Fusion：

$$
\operatorname{RRF}(d)
=
\sum_{r\in R}
\frac{1}{k+\operatorname{rank}_r(d)}
$$

$R$ 是多个排名列表，$k$ 是平滑常数。RRF 只依赖名次，不依赖不同系统的原始分数量纲。

### 5.1 RRF 手算

假设向量检索：

$$
d_3,d_1,d_2
$$

BM25：

$$
d_1,d_3,d_4
$$

取教学用小常数 $k=1$：

$$
\operatorname{RRF}(d_3)
=
\frac{1}{1+1}
+
\frac{1}{1+2}
=
\frac56
$$

$$
\operatorname{RRF}(d_1)
=
\frac{1}{1+2}
+
\frac{1}{1+1}
=
\frac56
$$

两者并列。真实系统常使用更大的平滑常数，并结合去重、metadata 与 reranker。

## 6. Recall@k

设某问题共有两个相关文档：

$$
G=\{d_2,d_5\}
$$

系统 top-3：

$$
R_3=[d_1,d_2,d_3]
$$

命中一个相关文档，因此：

$$
\operatorname{Recall@3}
=
\frac{|G\cap R_3|}{|G|}
=
\frac12
=0.5
$$

Recall 关注“相关证据有没有找回来”，不关心前 $k$ 中有多少噪声。

## 7. Precision@k

同一个例子：

$$
\operatorname{Precision@3}
=
\frac{|G\cap R_3|}{3}
=
\frac13
$$

Precision 关注返回结果的纯度。增大 $k$ 常提高 recall，却可能降低 precision 和增加 prompt 噪声。

## 8. MRR

若一组查询中，第一个相关结果的排名分别是：

$$
1,\quad2,\quad4
$$

倒数排名：

$$
1,\quad\frac12,\quad\frac14
$$

MRR：

$$
\operatorname{MRR}
=
\frac{1+\frac12+\frac14}{3}
\approx0.583
$$

MRR 强调第一个相关答案尽量靠前，适合用户只需要一个主要证据的场景。

## 9. nDCG 的直觉

当文档有“高度相关、部分相关、不相关”等多级标签时，可使用 DCG：

$$
\operatorname{DCG@k}
=
\sum_{i=1}^{k}
\frac{2^{rel_i}-1}
{\log_2(i+1)}
$$

高相关结果排名越靠前，贡献越大。再除以理想排序的 IDCG：

$$
\operatorname{nDCG@k}
=
\frac{\operatorname{DCG@k}}
{\operatorname{IDCG@k}}
$$

结果通常落在 $[0,1]$。

## 10. Reranker 的两阶段结构

第一阶段快速召回：

$$
N\text{ 个文档}
\longrightarrow
K_1\text{ 个候选}
$$

第二阶段使用更昂贵的 cross-encoder：

$$
K_1
\longrightarrow
K_2
$$

通常：

$$
K_2<K_1\ll N
$$

Embedding 将查询和文档分别编码，适合预计算；cross-encoder 同时读取查询和文档，交互更充分，但每对都要计算。

## 11. Chunk 大小如何实验

建立固定问题集后，对不同 chunk size 运行同样流程：

| Chunk | Recall@5 | 答案正确率 | 平均上下文 token | 延迟 |
|---|---:|---:|---:|---:|
| 128 | 待测 | 待测 | 待测 | 待测 |
| 256 | 待测 | 待测 | 待测 | 待测 |
| 512 | 待测 | 待测 | 待测 | 待测 |

可能出现：

- 小 chunk 定位准，但缺少解释所需上下文；
- 大 chunk 包含答案，但 embedding 主题混杂；
- overlap 提高边界召回，却带来重复上下文。

不存在适用于所有文档的神奇 chunk size。

## 12. Parent Document 的映射

索引小块：

$$
c_{1,1},c_{1,2},\ldots
$$

每个小块保存父文档 id：

$$
c_{1,2}\rightarrow p_1
$$

检索命中 $c_{1,2}$ 后返回更完整的父段落 $p_1$。若多个小块都指向 $p_1$，拼上下文前要按 parent id 去重。

## 13. Faithfulness 如何检查

把答案拆为原子声明：

$$
a=\{s_1,s_2,\ldots,s_m\}
$$

逐条判断检索证据是否支持。一个简单支持率：

$$
\operatorname{support}
=
\frac{
\#\text{有证据支持的声明}
}{
\#\text{全部可验证声明}
}
$$

它不能完全替代人工判断，但比只问“整段看起来是否正确”更容易定位幻觉。

## 14. 不可回答问题

评估集必须包含知识库没有答案的问题。系统应输出“资料不足”，而不是根据模型参数中的常识擅自回答。

需要分开评估：

- answerable 问题上的正确率；
- unanswerable 问题上的拒答率；
- 错误拒答率。

只提高拒答率很容易：所有问题都拒答即可。因此三者必须一起看。

## 15. 完整错误归因表

| 现象 | 检查 |
|---|---|
| 正确页完全没进索引 | 解析、过滤、权限、增量更新 |
| gold chunk 不在 top-k | embedding、query、chunk、索引 |
| 召回后被删掉 | fusion、reranker、去重、阈值 |
| 上下文有答案但回答错 | prompt、模型、冲突证据、截断 |
| 引用页码错 | metadata 绑定与解析页码 |
| A 用户看到 B 文档 | 检索前权限过滤缺失 |

每个失败案例先归类，再决定改哪个模块。
