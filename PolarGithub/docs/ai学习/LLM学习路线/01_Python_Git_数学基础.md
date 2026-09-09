# 模块 1：Python、数据处理与必要数学

## 目标

不是“学完 Python 全部语法”，而是掌握 LLM 项目反复出现的程序结构与数学语言：数据容器、函数、类、迭代器、文件格式、向量矩阵、概率、导数和梯度。

> 数学部分建议同时打开 [数学补充：从变化率到梯度下降](01A_数学直觉与手算.md)，跟着其中的数字完整手算一遍。

## 学习导航与动手方法

这一章不是让你“背 Python 语法表”，而是训练一种做 LLM 项目会反复用到的基本能力：

> 看到一段数据，能把它读进来；看到一个数组，能说出它的形状；看到一个公式，能把它翻译成代码；看到一个实验要求，知道先做最小版本，再逐步加功能。

如果你读懂了原理，但不知道作业从哪里下手，先别急着写完整答案。按照下面这个固定流程走：

1. **把题目改写成输入、处理、输出。**
   - 输入是什么？例如 JSONL 文件、NumPy 数组、随机生成的数据。
   - 处理是什么？例如清洗文本、计算均值、做 softmax、训练线性回归。
   - 输出是什么？例如统计表、形状检查结果、loss 曲线、实验结论。

2. **先造一个很小的假数据。**
   不要一上来处理大文件。先用 2 到 5 条样本，让你能肉眼看出结果对不对。

3. **先写能跑通的直线代码。**
   开始阶段可以先把所有逻辑写在一个文件里，从上往下执行。等跑通后，再拆成函数。

4. **每一步都打印中间结果。**
   新手最容易卡在“我以为它是这样，其实它不是”。所以要经常打印：
   - `type(x)`
   - `len(x)`
   - `x.shape`
   - 前几条样本 `x[:3]`
   - 概率和、loss、参数值等关键指标

5. **最后再整理成函数和报告。**
   先做出来，再变漂亮。推荐每个练习最终包含：
   - 一个 `.py` 脚本；
   - 一小段运行结果；
   - 一小段解释：我做了什么、验证了什么、遇到什么坑。

建议你在当前模块旁边建一个练习文件夹：

```text
module01_homework/
├── exercise_a_jsonl_stats.py
├── exercise_b_numpy_shapes.py
├── exercise_c_linear_regression.py
├── data/
│   └── demo.jsonl
└── notes.md
```

本章后半部分的“练习讲解”已经按这个思路拆成了可执行步骤。先跟着它完成最小版本；文末再独立完成课后 Project，不需要自己重新设计作业结构。

完整标准答案单独放在：[模块 1 综合练习标准答案](01_Python_Git_数学基础_综合练习标准答案.md)。建议你先自己按题目做一版，卡住时看提示，最后再对照标准答案。

## 1. Python 核心对象

### 变量与类型

变量是对象的名字，不是固定盒子。

```python
name = "Ada"       # str
age = 18           # int
loss = 2.31        # float
ready = True       # bool
missing = None     # NoneType
```

浮点数是有限精度近似，`0.1 + 0.2` 不保证精确等于 `0.3`。比较训练指标时用容差，而不是期待完全相等。

### list、tuple、dict、set

```python
tokens = [12, 7, 12]                    # 有序、可变
shape = (2, 3, 4)                       # 有序、通常表示固定结构
sample = {"text": "你好", "label": 1} # 键值映射
unique_ids = {12, 7}                    # 去重集合
```

数据样本常用 dict，batch 常是“字段到张量”的 dict。切片 `tokens[start:end]` 包含 start、不包含 end。

### 控制流

```python
valid = []
for item in samples:
    if len(item["text"].strip()) >= 2:
        valid.append(item)
```

先写清楚的循环，再学列表推导式。过度压缩代码会让新手更难观察中间值。

## 2. 函数：把数据流切成可测试单元

```python
def make_windows(token_ids: list[int], size: int) -> list[tuple[list[int], int]]:
    """用前 size 个 token 预测下一个 token。"""
    if size <= 0:
        raise ValueError("size must be positive")
    return [
        (token_ids[i:i + size], token_ids[i + size])
        for i in range(len(token_ids) - size)
    ]
```

好函数应有单一职责、清楚的输入输出和边界处理。类型标注帮助阅读，但运行时不会自动保证类型正确。

### 作用域与可变默认参数

不要这样写：

```python
def add_item(item, items=[]):  # 这个 list 会被多次调用共享
    items.append(item)
    return items
```

应写成：

```python
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

这讲的是 Python 里一个很经典的坑：**函数参数默认值不要用可变对象，比如 `[]`、`{}`、`set()`。**

你图里这个写法：

```python
def add_item(item, items=[]):
    items.append(item)
    return items
```

看起来意思是：

> 如果调用时没传 `items`，就默认用一个空列表。

但实际 Python 不是每次调用都新建一个空列表，而是：**函数定义的时候就创建了这一个 `[]`，之后每次调用都共用它。**

比如：

```python
def add_item(item, items=[]):
    items.append(item)
    return items

print(add_item("a"))
print(add_item("b"))
print(add_item("c"))
```

你可能以为输出是：

```
['a']
['b']
['c']
```

但实际是：

```
['a']
['a', 'b']
['a', 'b', 'c']
```

因为每次调用用的是**同一个默认列表**。

---

所以推荐这样写：

```python
def add_item(item, items=None):
    if items is None:
        items = []

    items.append(item)
    return items
```

这是什么意思呢？

```python
items=None
```

意思是：默认先不给列表，用 `None` 作为“没有传入列表”的标记。

然后：

```python
if items is None:
    items = []
```

意思是：如果调用者没有传 `items`，那就在这一次函数调用里面新建一个空列表。

这样每次调用都会得到独立的新列表。

例如：

```python
print(add_item("a"))
print(add_item("b"))
print(add_item("c"))
```

输出就是：

```
['a']
['b']
['c']
```

---

但如果你主动传入同一个列表，它还是会往那个列表里加：

```python
lst = [1, 2]

print(add_item(3, lst))
print(add_item(4, lst))
```

输出：

```
[1, 2, 3]
[1, 2, 3, 4]
```

这时候是合理的，因为你明确把 `lst` 传进去了。

核心记住一句话：

**Python 函数默认参数只在定义函数时创建一次，不是每次调用都重新创建。**
所以默认参数尽量用 `None`，不要直接写 `[]`。

## 3. 类和 `nn.Module` 的前置理解

类把状态和行为组织在一起。

```python
class Vocabulary:
    def __init__(self, tokens: list[str]):
        unique = sorted(set(tokens))
        self.token_to_id = {token: i for i, token in enumerate(unique)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id[ch] for ch in text]
```

`self` 表示当前实例。PyTorch 模型继承 `nn.Module`，参数作为对象状态，`forward` 定义数据如何通过模型。

## 4. 迭代器、生成器与 DataLoader

数据太大时不能一次读进内存。生成器按需产生数据：

```python
def read_nonempty_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                yield text
```

`yield` 暂停函数并保留状态。DataLoader 的思想类似：按 batch 取数据，还可打乱和并行读取。
这段主要在讲：**数据太大时，不要一次性全读进内存，而是用“生成器”一条一条读。**

核心代码是：

```python
def read_nonempty_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                yield text
```

它的作用是：**读取一个文本文件，逐行返回非空行。**

---

关键是 `yield`。

普通函数用 `return`：

```python
def f():
    return 1
```

调用后直接返回结果，函数结束。

但用了 `yield` 的函数就变成了**生成器函数**：

```python
def g():
    yield 1
    yield 2
    yield 3
```

调用它：

```python
x = g()
```

这时函数还没有真正执行，只是得到了一个“生成器对象”。

只有你开始取数据时，它才会一点一点执行：

```python
for v in x:
    print(v)
```

输出：

```
1
2
3
```

---

你图里的函数可以这样用：

```python
for text in read_nonempty_lines("data.txt"):
    print(text)
```

它不是先把整个 `data.txt` 读完，而是：

```
读一行
处理一行
yield 返回一行
暂停
下次继续读下一行
```

所以内存占用很小。

---

对比一下。

如果你这样写：

```python
def read_nonempty_lines(path):
    result = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                result.append(text)

    return result
```

这会把所有非空行都存到 `result` 里。

如果文件很大，比如 10GB，就可能爆内存。

但生成器版本：

```python
def read_nonempty_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                yield text
```

不会一次性存所有内容，而是需要一条给一条。

---

`yield` 还有一个很重要的特性：**暂停函数，并保留状态。**

比如：

```python
def count():
    print("开始")
    yield 1
    print("继续")
    yield 2
    print("结束")
```

运行：

```python
g = count()

print(next(g))
print(next(g))
```

过程是：

第一次 `next(g)`：

```
开始
1
```

函数执行到 `yield 1` 就暂停。

第二次 `next(g)`：

```
继续
2
```

它会从上次暂停的位置继续执行。

所以图里说：

> `yield` 暂停函数并保留状态。

意思就是：函数不会死掉，它会记住自己读到文件哪一行了，下次继续往后读。

---

最后一句说 DataLoader：

> DataLoader 的思想类似：按 batch 取数据，还可打乱和并行读取。

比如 PyTorch 训练模型时，数据集可能很大，不能一次性全部塞进显存/内存。

所以 DataLoader 会一批一批给模型：

```python
for batch_x, batch_y in dataloader:
    output = model(batch_x)
```

它的思想和生成器很像：

```
需要一批数据
就读取一批
处理一批
再读取下一批
```

而不是：

```
一开始把全部数据读完
```

所以你可以把它理解成：

**生成器是 Python 层面的“按需吐数据”；DataLoader 是深度学习里更高级的“按 batch 按需吐数据”。**
这段代码的功能是：

> 打开一个文本文件，逐行读取，把**非空行**去掉首尾空白后，一个一个“吐出来”。

代码：

```python
def read_nonempty_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                yield text
```

逐项拆开看。

---

### 1. `def read_nonempty_lines(path):`

```python
def read_nonempty_lines(path):
```

定义一个函数，名字叫：

```python
read_nonempty_lines
```

意思大概是“读取非空行”。

括号里的：

```python
path
```

是参数，表示**文件路径**。

比如你之后可以这样调用：

```python
read_nonempty_lines("data.txt")
```

这里 `"data.txt"` 就会传给 `path`。

也就是说，函数里面的：

```python
path
```

此时就代表：

```python
"data.txt"
```

---

### 2. `with open(path, "r", encoding="utf-8") as f:`

```python
with open(path, "r", encoding="utf-8") as f:
```

这句是在**打开文件**。

拆开：

```python
open(path, "r", encoding="utf-8")
```

意思是：

> 按照 `utf-8` 编码，以只读模式打开 `path` 指向的文件。

其中：

```python
"path"
```

不是固定的文件名，而是前面传进来的参数。

比如：

```python
read_nonempty_lines("data.txt")
```

那这里就相当于：

```python
open("data.txt", "r", encoding="utf-8")
```

---

这里的 `"r"` 是 read 的意思，表示**读取模式**。

常见模式有：

```python
"r"   # 读文件
"w"   # 写文件，会覆盖原内容
"a"   # 追加写入
```

---

`encoding="utf-8"` 是指定文本编码。

很多中文文本都用 `utf-8`，所以一般写上比较稳。

---

`as f` 的意思是：

> 把打开的文件对象临时命名为 `f`。

所以后面：

```python
for line in f:
```

就是在遍历这个文件对象。

---

`with` 的作用是：

> 文件用完后自动关闭。

如果不用 `with`，传统写法是：

```python
f = open(path, "r", encoding="utf-8")
# 读取文件
f.close()
```

但是如果中途报错，可能忘记关闭文件。

所以更推荐：

```python
with open(...) as f:
    ...
```

它会自动帮你关闭文件。

---

### 3. `for line in f:`

```python
for line in f:
```

这句是在**逐行遍历文件**。

文件对象 `f` 可以被循环，每次循环取出文件的一行。

比如文件内容是：

```
hello

world
python
```

那么：

```python
for line in f:
```

会依次得到：

```python
"hello\n"
"\n"
"world\n"
"python"
```

注意，读出来的每一行通常会带有换行符 `\n`。

---

### 4. `text = line.strip()`

```python
text = line.strip()
```

这句是把当前这一行的**首尾空白字符去掉**。

`strip()` 会去掉：

```
空格
换行符 \n
制表符 \t
```

比如：

```python
line = "   hello world\n"
text = line.strip()
```

结果是：

```python
"hello world"
```

再比如：

```python
line = "\n"
text = line.strip()
```

结果是：

```python
""
```

也就是空字符串。

所以这句的作用是：

> 把一行清理干净，去掉前后的空格和换行。

---

### 5. `if text:`

```python
if text:
```

这句是在判断：

> `text` 是否不是空字符串。

在 Python 里，空字符串：

```python
""
```

会被当成 `False`。

非空字符串：

```python
"hello"
```

会被当成 `True`。

所以：

```python
if text:
```

等价于：

```python
if text != "":
```

但 Python 里更常写成：

```python
if text:
```

比如：

```python
text = ""
if text:
    print("非空")
```

不会执行。

而：

```python
text = "hello"
if text:
    print("非空")
```

会执行。

所以这里的意思是：

> 如果这一行去掉空白后还有内容，就继续处理；如果是空行，就跳过。

---

### 6. `yield text`

```python
yield text
```

这是最关键的一句。

它不是普通的 `return`，而是 `yield`。

`yield` 的意思是：

> 先把 `text` 交出去，然后函数暂停；下一次需要数据时，再从这里继续往下执行。

因此这个函数不是一次性返回一个列表，而是变成了一个**生成器**。

---

比如有一个文件 `data.txt`：

```
apple

banana


cat
```

调用：

```python
for x in read_nonempty_lines("data.txt"):
    print(x)
```

输出是：

```
apple
banana
cat
```

中间空行会被跳过。

---

这段代码的执行过程大概是：

```
打开文件
读第 1 行
去掉首尾空白
如果不是空行，就 yield 出去

暂停

下一次循环需要数据
继续读第 2 行
如果是空行，跳过

继续读第 3 行
yield 出去

……
```

---

一个非常重要的点是：

```python
read_nonempty_lines("data.txt")
```

这句本身不会立刻把文件全部读完。

它只是创建了一个生成器对象。

真正读取发生在你开始遍历它的时候：

```python
for line in read_nonempty_lines("data.txt"):
    print(line)
```

或者手动取：

```python
g = read_nonempty_lines("data.txt")

print(next(g))
print(next(g))
```

---

所以这段代码可以理解成：

```python
def read_nonempty_lines(path):        # 定义一个函数，参数是文件路径
    with open(path, "r", encoding="utf-8") as f:   # 打开文件，命名为 f
        for line in f:                # 一行一行读取文件
            text = line.strip()       # 去掉这一行前后的空白和换行符
            if text:                  # 如果去掉后不是空字符串
                yield text            # 把这一行交出去，然后暂停
```

核心记住：

**`return` 是一次性返回结果，函数结束；`yield` 是返回一个结果后暂停，下次还能继续。**

因为这一句本身就在**定义/赋值** `line`：

```python
for line in f:
```

它的意思不是“使用一个已经定义好的 `line`”，而是：

> 每次从 `f` 里面取出一个东西，把它临时命名为 `line`。

所以 `line` 是在 `for` 循环中自动产生的变量。

比如：

```python
for x in [10, 20, 30]:
    print(x)
```

这里 `x` 之前也没有定义，但 Python 会自动让：

```
第一次循环：x = 10
第二次循环：x = 20
第三次循环：x = 30
```

同理：

```python
for line in f:
```

就是：

```
第一次循环：line = 文件第 1 行
第二次循环：line = 文件第 2 行
第三次循环：line = 文件第 3 行
……
```

---

为什么 `line` 是“行”？不是因为名字叫 `line`，而是因为 **`f` 是文件对象，文件对象被 `for` 遍历时，每次默认吐出一行文本**。

也就是说，关键是右边的 `f`：

```python
with open(path, "r", encoding="utf-8") as f:
```

这里 `f` 是打开后的文件对象。

文件对象有一个特性：

```python
for line in f:
```

会自动一行一行读取文件。

所以你写成这样也可以：

```python
for x in f:
    text = x.strip()
```

这里 `x` 也是一行。

甚至写成：

```python
for abc in f:
    text = abc.strip()
```

也可以。

只是一般为了可读性，大家写成：

```python
for line in f:
```

因为它确实代表“当前这一行”。

---

所以这句：

```python
for line in f:
```

可以理解成更直白的伪代码：

```python
每次从文件 f 里读取一行
把这一行放进变量 line
然后执行下面缩进的代码
```

完整逻辑就是：

```python
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        text = line.strip()
```

等价理解为：

```
打开文件，叫它 f
从 f 里取第一行，赋值给 line
把 line 去掉首尾空白，赋值给 text

再从 f 里取第二行，赋值给 line
把 line 去掉首尾空白，赋值给 text

再从 f 里取第三行……
```

---

Python 和 C 不太一样，变量通常不需要提前声明。

C 里你可能要写：

```c
int x;
x = 10;
```

但 Python 里直接：

```python
x = 10
```

变量就产生了。

所以：

```python
for line in f:
```

里的 `line` 也是在循环开始时由 Python 自动赋值产生的。

## 5. 文件与常见数据格式

- TXT：简单语料。
- CSV：表格；文本中可能含逗号和换行，应使用库而非手工 `split(',')`。
- JSON：一个完整的嵌套对象。
- JSONL：每行一个 JSON，适合流式读取和训练数据。
- Parquet：带类型的列式格式，适合大数据。

JSONL 示例：

```json
{"instruction":"计算 2+3","output":"5"}
{"instruction":"解释勾股定理","output":"在直角三角形中……"}
```

读取时明确编码，并把不可信字段当作数据，不用 `eval()`。

## 6. NumPy：数组不只是更快的 list

NumPy 数组通常是同一数据类型的规则网格。

```python
import numpy as np

x = np.array([[1.0, 2.0], [3.0, 4.0]])
w = np.array([[0.1, -0.2], [0.3, 0.4]])
y = x @ w
print(x.shape, w.shape, y.shape)  # (2,2) (2,2) (2,2)
```

LLM 学习中最重要的习惯是给每个维度命名：

- `B`：batch size，一次处理多少样本。
- `T`：sequence length，每个样本多少 token。
- `C` 或 `D`：embedding/model dimension，每个 token 的向量长度。
- `V`：vocabulary size。

例如隐藏状态 `x` 为 `[B, T, C]`，词表 logits 为 `[B, T, V]`。

### 广播

```python
x = np.zeros((2, 3, 4))
bias = np.ones((4,))
y = x + bias  # bias 被视为 [1,1,4]，沿前两维广播
```

广播很方便，也可能静默制造错误。写代码前把形状写在注释中。

## 7. 线性代数的直觉

### 标量、向量、矩阵、张量

标量是一个数；向量是一列特征；矩阵可看作多个向量或线性变换；深度学习里“张量”泛指多维数组。

线性层：

$$
Y = XW + b
$$

若 `X:[B,in]`、`W:[in,out]`、`b:[out]`，则 `Y:[B,out]`。`W` 的每一列可以理解为一个输出特征的检测方向。

### 点积和相似度

$$
x \cdot y = \sum_i x_i y_i
$$

点积同时受方向和长度影响。余弦相似度消除长度影响：

$$
\cos(x,y)=\frac{x\cdot y}{\|x\|\|y\|}
$$

Embedding 检索经常用点积或余弦相似度。是否先归一化会改变二者关系。

### 转置

`X:[m,n]` 的转置 `X.T:[n,m]`。Attention 中 `Q @ K.transpose(-2,-1)` 是让每个 query 和每个 key 做点积。
这页核心在讲：**NumPy 数组不是“更快的 Python list”，而是有形状、有维度、通常同类型的数学数组。**

---

## 7A. NumPy 逐句详解：导入、形状、矩阵乘法与广播

这一段是对第 6 节 NumPy 代码的逐句拆解。它保留在这里，是为了让你在看完“线性代数直觉”后，再回头把 NumPy 的形状、矩阵乘法和广播真正落到代码上。

### 7A.1 `import numpy as np`

```python
import numpy as np
```

导入 NumPy，并起别名叫 `np`。

以后写：

```python
np.array(...)
np.zeros(...)
np.ones(...)
```

不用每次写完整的：

```python
numpy.array(...)
```

---

### 7A.2 创建二维数组

```python
x = np.array([[1.0, 2.0], [3.0, 4.0]])
w = np.array([[0.1, -0.2], [0.3, 0.4]])
```

这里 `x` 是一个二维数组，可以看成矩阵：

```
x =
[[1.0, 2.0],
 [3.0, 4.0]]
```

`w` 也是一个二维数组：

```
w =
[[0.1, -0.2],
 [0.3,  0.4]]
```

注意，NumPy 数组和普通 list 不一样。

普通 list：

```python
a = [[1.0, 2.0], [3.0, 4.0]]
```

本质上是“列表里面套列表”。

NumPy 数组：

```python
x = np.array([[1.0, 2.0], [3.0, 4.0]])
```

更像一个规则网格：

```
2 行 × 2 列
```

它有明确的形状、数据类型，并且可以做矩阵运算。

---

### 7A.3 `@` 是矩阵乘法

```python
y = x @ w
```

这里的 `@` 表示矩阵乘法。

也就是：

```
y = xw
```

具体算：

```
x =
[[1, 2],
 [3, 4]]

w =
[[0.1, -0.2],
 [0.3,  0.4]]
```

矩阵乘法结果：

```
y[0,0] = 1×0.1 + 2×0.3 = 0.7
y[0,1] = 1×(-0.2) + 2×0.4 = 0.6

y[1,0] = 3×0.1 + 4×0.3 = 1.5
y[1,1] = 3×(-0.2) + 4×0.4 = 1.0
```

所以：

```python
y =
[[0.7, 0.6],
 [1.5, 1.0]]
```

---

### 7A.4 `.shape` 是数组形状

```python
print(x.shape, w.shape, y.shape)
```

输出：

```python
(2, 2) (2, 2) (2, 2)
```

意思是：

```
x 是 2 行 2 列
w 是 2 行 2 列
y 是 2 行 2 列
```

`.shape` 返回的是一个元组。

比如：

```python
x.shape
```

得到：

```python
(2, 2)
```

第一个 `2` 是行数，第二个 `2` 是列数。

---

### 7A.5 LLM 里常用的维度命名

这一页说：

```
B: batch size
T: sequence length
C 或 D: embedding/model dimension
V: vocabulary size
```

这些是深度学习里很常见的维度符号。

比如：

```python
x.shape = [B, T, C]
```

意思是：

```
B 个样本
每个样本有 T 个 token
每个 token 用 C 维向量表示
```

举个例子：

```python
x.shape = [2, 3, 4]
```

可以理解成：

```
一次处理 2 句话
每句话 3 个 token
每个 token 是 4 维向量
```

所以它不是二维矩阵，而是三维数组。

---

### 7A.6 hidden state `[B, T, C]`

图里说：

```
隐藏状态 x 为 [B, T, C]
```

比如：

```python
x.shape = [2, 3, 4]
```

意思是：

```
2 个句子
每个句子 3 个 token
每个 token 有 4 个隐藏特征
```

可以想象成：

```
第 1 个句子：
    第 1 个 token: 4 维向量
    第 2 个 token: 4 维向量
    第 3 个 token: 4 维向量

第 2 个句子：
    第 1 个 token: 4 维向量
    第 2 个 token: 4 维向量
    第 3 个 token: 4 维向量
```

---

### 7A.7 logits `[B, T, V]`

图里又说：

```
词表 logits 为 [B, T, V]
```

这里 `V` 是 vocabulary size，也就是词表大小。

比如词表里有 50000 个 token，那么：

```python
logits.shape = [B, T, 50000]
```

意思是：

```
对每个样本
对每个位置
模型都输出一个长度为 50000 的向量
```

这个向量表示：

> 当前位置下一个 token 是词表中每个 token 的分数。

比如：

```
logits[b, t]
```

就是第 `b` 个样本、第 `t` 个位置，对整个词表的预测分数。

---

### 7A.8 广播 broadcasting

下面这段：

```python
x = np.zeros((2, 3, 4))
bias = np.ones((4,))
y = x + bias
```

先看：

```python
x = np.zeros((2, 3, 4))
```

创建一个全 0 数组，形状是：

```
(2, 3, 4)
```

也就是：

```
2 层
每层 3 行
每行 4 个数
```

再看：

```python
bias = np.ones((4,))
```

创建一个长度为 4 的一维数组：

```python
bias = [1, 1, 1, 1]
```

它的形状是：

```
(4,)
```

然后：

```python
y = x + bias
```

问题来了：

```
x.shape    = (2, 3, 4)
bias.shape =       (4,)
```

它们形状不一样，为什么可以相加？

因为 NumPy 会进行**广播**。

它会把：

```python
bias.shape = (4,)
```

理解成：

```python
bias.shape = (1, 1, 4)
```

然后沿前两个维度复制使用：

```
x:    (2, 3, 4)
bias: (1, 1, 4)
```

于是 `bias` 会加到 `x` 的最后一维上。

也就是说，每一个长度为 4 的向量都会加上：

```python
[1, 1, 1, 1]
```

所以：

```python
x = np.zeros((2, 3, 4))
bias = np.ones((4,))
y = x + bias
```

结果 `y` 的形状还是：

```python
(2, 3, 4)
```

并且里面每个数都变成了 `1`。

---

#### 7A.8.1 广播的正式判断规则：从右往左对齐

广播不是看“两个数组相差几维”，而是把两个 `shape` **右对齐**，从最右边向左逐项检查。每一对维度满足下面任一条件就兼容：

1. 两个数字相等；
2. 其中一个数字是 `1`；
3. 某个数组缺少这一维时，把它看成左边补了 `1`。

例如：

```text
(4, 2, 3)
      (3,)
```

右对齐并在左边补 `1`：

```text
(4, 2, 3)
(1, 1, 3)
```

三组维度分别是“4 与 1、2 与 1、3 与 3”，全部兼容，所以结果形状是 `(4, 2, 3)`。由此可见，维度数量相差 2 仍然可以广播。

相反：

```text
(2, 3)
   (2,)
```

右对齐后最后一维是 `3` 与 `2`，既不相等，也没有一个是 `1`，因此不能广播。维度数量只差 1 也不一定成功。

#### 7A.8.2 `(3,)`、`(1,3)` 和 `(3,1)` 不是一回事

```text
(3,)    一维数组，长度为 3
(1, 3)  二维数组，1 行 3 列
(3, 1)  二维数组，3 行 1 列
```

一维数组 `(3,)` 在广播时从右边对齐，因此经常表现得像 `(1,3)`，但它本身没有明确的“行”或“列”方向。

假设：

```python
a = np.array([
    [1, 2, 3],
    [4, 5, 6],
])                         # a.shape == (2, 3)
b = np.array([10, 20, 30]) # b.shape == (3,)
```

广播比较相当于：

```text
(2, 3)
(1, 3)
```

所以 `a + b` 会给每一行都加上 `[10,20,30]`。

#### 7A.8.3 想给每一行加不同的数，要显式制造列维度

如果想让第一行整体加 10、第二行整体加 20，`(2,)` 不够，因为它会与最后一维对齐。需要把它改成 `(2,1)`：

```python
row_bias = np.array([10, 20])[:, None]
# 等价写法：
# row_bias = np.array([10, 20]).reshape(2, 1)

result = a + row_bias
```

形状比较为：

```text
a:        (2, 3)
row_bias: (2, 1)
result:   (2, 3)
```

长度为 1 的最后一维会扩展到 3，因此结果是：

```text
[[11, 12, 13],
 [24, 25, 26]]
```

#### 7A.8.4 高维广播和结果形状

只要每个对齐维度都兼容，高维数组也遵守同一条规则。例如：

```text
(2, 2, 2)
(1, 2, 1)
```

从右向左比较是“2 与 1、2 与 2、2 与 1”，所以可以广播，结果为 `(2,2,2)`。

再例如：

```text
(4, 1, 3)
(1, 5, 1)
```

每一维取广播后的有效大小：

```text
4 与 1 → 4
1 与 5 → 5
3 与 1 → 3
```

结果形状是 `(4,5,3)`。这里所谓“取较大值”只适用于已经确认兼容的维度；如果是 `3` 与 `2`，不能直接取 3，而是立即报错。

#### 7A.8.5 在线性层和 Softmax 中为什么经常遇到广播

线性层写成：

```python
z = x @ W + b
```

若：

```text
x.shape     = (B, D_in)
W.shape     = (D_in, D_out)
(x @ W).shape = (B, D_out)
b.shape     = (D_out,)
```

偏置 `b` 会被看成 `(1,D_out)`，于是同一个偏置向量被加到 batch 中每个样本上。它不会在不同样本之间做矩阵乘法。

Softmax 中常见：

```python
row_max = np.max(x, axis=1, keepdims=True)
shifted = x - row_max
```

如果 `x.shape == (B,C)`，使用 `keepdims=True` 后 `row_max.shape == (B,1)`，正好能让每个样本的所有类别减去该样本自己的最大值。若去掉 `keepdims=True`，结果是 `(B,)`，它会尝试和最后一维 `C` 对齐，通常无法广播或产生错误含义。

#### 7A.8.6 广播检查步骤

看到两个不同形状的数组做逐元素运算时，按这个顺序检查：

1. 写出两个完整 `shape`。
2. 从右向左对齐，维度少的一方在左边补 `1`。
3. 每一对必须相等，或者其中一个是 `1`。
4. 写出结果形状。
5. 用一句话说明“哪个轴被重复使用”，确认这真是你的意图。

完整的更多例子仍保留在：[NumPy 广播机制整理](../numpy广播机制.md)。

### 7A.9 广播的直觉

这句：

```python
y = x + bias
```

可以想象成：

```
对 x 里的每个长度为 4 的小向量，都加上 bias
```

比如：

```
[0, 0, 0, 0] + [1, 1, 1, 1] = [1, 1, 1, 1]
```

重复很多次。

---

### 7A.10 为什么说广播可能“静默制造错误”

因为有时候代码不会报错，但结果不是你想要的。

比如你本来想让：

```
每个 batch 加一个 bias
```

但 NumPy 可能按照最后一维去广播。

所以写深度学习代码时，经常要在旁边注释形状：

```python
# x: [B, T, C]
# bias: [C]
y = x + bias
```

这样你能清楚知道：

> 这个 bias 是加到每个 token 的 C 维特征上的。

这就是图里最后一句：

```
写代码前把形状写在注释中。
```

的意思。

---

这页最重要的三句话：

```
NumPy 数组有 shape，不是普通 list。
@ 表示矩阵乘法。
广播会自动扩展维度，但要小心形状是否符合你的本意。
```

## 8. 概率与信息论

### 概率分布

分类模型输出每个类别的概率，所有概率非负且和为 1。语言模型的类别就是词表中的下一个 token。

Softmax 把任意 logits 转成概率：

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

数值实现时先减去最大 logit，避免指数溢出：

```python
def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)
```

### 交叉熵

真实答案为类别 `y` 时，单样本损失是：

$$
L=-\log p_y
$$

模型给正确答案高概率，损失小；给正确答案接近 0 的概率，损失巨大。交叉熵不是“错误率”，它还衡量模型有多自信。

### 困惑度

语言模型常用 `perplexity = exp(cross_entropy)`。可粗略理解为模型平均在多少个等可能选择间犹豫。它只能在相同 tokenization 和数据处理下公平比较。

这页讲的是：**模型怎么把“分数”变成“概率”，以及怎么衡量模型猜得好不好。**

你可以把语言模型想象成在做选择题：

> 给定前面的文本，模型要从词表里选出“下一个 token”。

比如：

```
我今天去食堂吃了一个 [ ? ]
```

词表里可能有：

```
苹果、篮球、微积分、电脑、包子……
```

模型要给每个候选 token 一个分数，然后转成概率。

---

### 概率补充 1：logits 是“原始分数”

模型一开始输出的不是概率，而是一堆原始分数，叫 **logits**。

比如模型对三个词的打分是：

```python
logits = [2.0, 1.0, 0.1]
```

可以理解成：

```
苹果：2.0
包子：1.0
篮球：0.1
```

分数越大，模型越倾向于选它。

但 logits 不是概率，因为它们：

```
可以是负数
不一定加起来等于 1
没有概率意义
```

所以要用 softmax 把它们变成概率。

---

### 概率补充 2：Softmax 把 logits 变成概率

公式是：

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

这里：

```
z_i 是第 i 个类别的 logit
p_i 是第 i 个类别的概率
```

比如 logits 是：

```python
[2.0, 1.0, 0.1]
```

softmax 之后大概会变成：

```python
[0.659, 0.242, 0.099]
```

意思就是：

```
第一个词概率 65.9%
第二个词概率 24.2%
第三个词概率 9.9%
```

所有概率相加等于 1。

---

### 概率补充 3：为什么要用指数 $e^x$？

因为指数函数有几个好处：

第一，它永远是正数：

$$
e^x > 0
$$

所以 softmax 得到的概率不会是负数。

第二，分数差距会被放大。

比如：

```
logits: 2, 1, 0
```

差距看起来只是 1。

但指数之后：

```
e^2 ≈ 7.39
e^1 ≈ 2.72
e^0 = 1
```

大的分数会变得更突出。

这符合分类直觉：

> 模型觉得某个词明显更合适时，它的概率应该明显更高。

---

### 概率补充 4：为什么代码里要先减去最大值？

图里代码是：

```python
def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)
```

数学上，softmax 对所有 logits 同时减去同一个数，结果不变。

比如：

$$
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

如果所有 $z$ 都减去最大值 $m$，变成：

$$
\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}
$$

因为：

$$
e^{z_i-m} = \frac{e^{z_i}}{e^m}
$$

分子分母都会除以 $e^m$，所以结果不变。

---

为什么要这样做？

为了防止指数爆炸。

比如：

```python
np.exp(1000)
```

这个数巨大无比，计算机会溢出。

所以先减去最大值：

```python
[1000, 999, 998]
```

变成：

```python
[0, -1, -2]
```

这样再算指数：

```python
e^0, e^-1, e^-2
```

就安全很多。

所以这句：

```python
shifted = x - np.max(x, axis=axis, keepdims=True)
```

不是改变结果，而是为了**数值稳定**。

---

### 概率补充 5：`axis=-1` 是什么意思？

```python
axis=-1
```

表示沿着**最后一个维度**做 softmax。

在语言模型里，logits 通常形状是：

```python
[B, T, V]
```

其中：

```
B：batch size，多少个样本
T：sequence length，每个样本多少个 token
V：vocabulary size，词表大小
```

比如：

```python
logits.shape = [2, 3, 50000]
```

意思是：

```
2 个句子
每个句子 3 个位置
每个位置对 50000 个词打分
```

所以 softmax 应该在最后一维 `V` 上做。

也就是：

```
对每个位置，把 50000 个词的分数转成概率
```

所以写：

```python
axis=-1
```

---

### 概率补充 6：交叉熵是什么？

图里写：

$$
L = -\log p_y
$$

这里 $y$ 是真实答案类别。

比如真实答案是“包子”。

模型输出概率：

```
苹果：0.7
包子：0.2
篮球：0.1
```

那么正确答案“包子”的概率是：

$$
p_y = 0.2
$$

损失就是：

$$
L = -\log 0.2
$$

这个值比较大，说明模型不太相信正确答案。

如果模型输出：

```
苹果：0.05
包子：0.9
篮球：0.05
```

那么：

$$
p_y = 0.9
$$

损失是：

$$
L = -\log 0.9
$$

这个值很小，说明模型很相信正确答案。

所以交叉熵的直觉是：

> 正确答案概率越高，损失越小；正确答案概率越低，损失越大。

---

### 概率补充 7：为什么不是直接看“对没对”？

比如真实答案是 B。

模型 1：

```
A: 0.49
B: 0.51
C: 0.00
```

模型 2：

```
A: 0.01
B: 0.98
C: 0.01
```

两个模型都选 B，都“答对了”。

但模型 2 显然更自信、更好。

交叉熵可以区分这两种情况。

所以图里说：

> 交叉熵不是“错误率”，它还衡量模型有多自信。

---

### 概率补充 8：困惑度 perplexity 是什么？

图里写：

```
perplexity = exp(cross_entropy)
```

也就是：

$$
\text{perplexity} = e^{\text{cross entropy}}
$$

它可以粗略理解为：

> 模型平均在多少个差不多可能的选择中犹豫。

比如：

```
perplexity = 10
```

可以粗略理解为：

> 模型平均每一步像是在 10 个差不多可能的 token 里选择。

如果：

```
perplexity = 100
```

说明模型更困惑。

所以困惑度越低，一般说明语言模型越好。

---

但注意，困惑度不能随便跨模型比较。

图里说：

```
只能在相同 tokenization 和数据处理下公平比较
```

这是因为不同模型切词方式不同。

比如同一句话：

```
我喜欢机械设计
```

一个 tokenizer 可能切成：

```
我 / 喜欢 / 机械 / 设计
```

另一个可能切成：

```
我 / 喜 / 欢 / 机 / 械 / 设 / 计
```

token 数不同，计算出来的平均损失和困惑度就不可直接比较。

---

这页最核心的链条是：

```
logits 原始分数
    ↓ softmax
概率分布
    ↓ 看真实答案的概率
交叉熵 loss
    ↓ exp(loss)
困惑度 perplexity
```

一句话总结：

**softmax 负责把模型输出的分数变成概率；交叉熵负责惩罚“正确答案概率太低”；困惑度是语言模型常用的整体难度指标。**

## 9. 导数、梯度与链式法则

导数表示输入微小变化时输出如何变化。多参数函数的梯度是所有偏导数组成的向量，指向函数上升最快方向；因此最小化损失时沿负梯度更新：

$$
\theta \leftarrow \theta-\eta\nabla_\theta L
$$

其中 `η` 是学习率。太大可能越过低点甚至发散；太小则训练缓慢。

链式法则说明复合函数的导数如何相乘：若 `L=f(g(x))`，则

$$
\frac{dL}{dx}=\frac{dL}{dg}\frac{dg}{dx}
$$

反向传播就是从损失开始，沿计算图反向反复应用链式法则。

### 数值梯度检查

$$
f'(x)\approx\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}
$$

数值梯度慢但直观，适合检查你手写的反向传播。`ε` 不能过大，也不能小到被浮点误差淹没，常从 `1e-5` 尝试。

## 10. 统计与实验基础

训练集用于学习参数；验证集用于选择超参数和早停；测试集只在最终评价时使用。反复看测试结果并据此调参，相当于把测试集偷偷变成验证集。

必须理解：

- 均值容易受极端值影响；同时看中位数和分位数。
- 相关不等于因果。
- 单次随机种子的提升可能是运气；重要实验运行多个 seed。
- 只改一个变量，才能判断变化来自哪里。
- 数据泄漏会让指标虚高，例如同一文档切出的相邻块分别进入训练和测试。

## 11. 练习讲解：从读题到动手

### 11.1 解题提示与逐步讲解

如果你“看懂了，但不知道怎么下手”，先记住一个原则：

> 作业不是一开始就写完整大程序，而是把题目拆成一个个小检查点。每完成一个检查点，就运行一次、打印一次、确认一次。

建议你按这个顺序做三个练习：

1. 先做练习 A，因为它主要训练 Python 基本功：读文件、循环、字典、计数。
2. 再做练习 B，因为它训练 NumPy 形状意识：`shape`、`reshape`、矩阵乘法、softmax。
3. 最后做练习 C，因为它把数学、NumPy 和实验记录合在一起。

每个练习都建议保存一次 Git：

```bash
git add .
git commit -m "finish module01 exercise A"
```

如果你现在还不熟 Git，也可以先简单理解成“给当前作业拍一张快照”。后面出错了，至少知道自己完成过哪个阶段。

### 练习 A：纯 Python 文本分析

读取 JSONL，对文本长度分桶，统计空样本、重复样本和高频字符；不能用 pandas。

#### A.1 先把题目翻译成人话

这个作业不是考你会不会“高级数据分析”，而是考你能不能用纯 Python 完成一个小型数据清洗流程。

输入：

- 一个 `.jsonl` 文件；
- 每一行应该是一个 JSON 对象；
- 每个对象里通常有文本字段，比如 `text`、`instruction`、`output`。

处理：

- 逐行读取；
- 跳过空行；
- 尝试把每行解析成 JSON；
- 找到文本内容；
- 统计文本长度；
- 记录空样本；
- 记录重复样本；
- 统计高频字符。

输出：

- 总行数；
- 成功解析多少行；
- 坏行多少行；
- 空文本多少条；
- 重复文本多少条；
- 不同长度区间各有多少条；
- 出现最多的若干字符。
  JSON 文件是一种用来**保存结构化数据**的纯文本文件，扩展名通常是 `.json`。

例如：

```json
{
  "name": "Polar",
  "age": 20,
  "isStudent": true,
  "skills": ["Python", "CAD", "机械工程"]
}
```

它可以理解成 Python 字典和列表写进文件后的样子：

```python
data = {
    "name": "Polar",
    "age": 20,
    "isStudent": True,
    "skills": ["Python", "CAD", "机械工程"]
}
```

JSON 常见规则：

- 对象用 `{}`，类似 Python 字典
- 数组用 `[]`，类似 Python 列表
- 键必须使用双引号
- 字符串也必须使用双引号
- 布尔值写成 `true`、`false`
- 空值写成 `null`
- 不能直接写注释

Python 读取 JSON 文件：

```python
import json

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(data["name"])
```

Python 写入 JSON 文件：

```python
import json

data = {
    "name": "Polar",
    "age": 20
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
```

其中：

- `ensure_ascii=False`：中文不会变成转义字符
- `indent=4`：自动缩进，让文件更容易阅读

JSON 经常用于保存软件设置、网页接口数据、游戏存档、模型配置和项目参数。
因为 `f` 和 `data` 代表的东西不一样。

```python
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
```

其中：

```python
f
```

是**文件对象**，表示“已经打开的这个文件”，它本身不是最终的数据。

而：

```python
data = json.load(f)
```

是让 `json.load` 从 `f` 里读取内容，并把 JSON 转换成 Python 数据，然后保存到 `data`。

例如文件内容是：

```json
{
  "name": "Polar",
  "age": 20
}
```

那么：

```python
f
```

类似于“这本书本身”；

```python
data
```

类似于“从书里读出来并整理好的内容”：

```python
{
    "name": "Polar",
    "age": 20
}
```

所以才能写：

```python
print(data["name"])
```

不能直接写：

```python
print(f["name"])
```

因为 `f` 是文件对象，不是字典。

完整流程是：

```
data.json 文件
    ↓ open
f：文件对象
    ↓ json.load(f)
data：Python 字典
```

而且离开 `with` 后，`f` 会被关闭，但 `data` 仍然可以继续使用：

```python
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 此时 f 已关闭，但 data 还在
print(data["name"])
```

#### A.2 先造一个最小 JSONL 文件

不要一开始就找大数据集。先在 `module01_homework/data/demo.jsonl` 里写几行：

```jsonl
{"text": "hello world"}
{"text": ""}
{"text": "机器学习"}
{"text": "hello world"}
这不是合法 JSON
{"instruction": "解释 AI", "output": "AI 是人工智能。"}
```

这个小文件故意包含：

- 正常文本；
- 空文本；
- 重复文本；
- 坏行；
- 没有 `text`，但有 `instruction` 和 `output` 的样本。

这样你的程序如果能处理它，就说明基本逻辑是对的。

#### A.3 第一版先只完成“能读每一行”

先新建 `exercise_a_jsonl_stats.py`，写最小版本：

```python
from pathlib import Path

path = Path("data/demo.jsonl")

with path.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        print(line_no, repr(line))
```

先运行它：

```bash
python exercise_a_jsonl_stats.py
```

你要先确认：程序真的能读到每一行。不要跳过这一步。很多新手不是卡在算法，而是文件路径一开始就错了。
这几句的意思是：**用 `pathlib` 表示一个文件路径，然后以只读方式打开这个文件。**

```python
from pathlib import Path
```

从 Python 的 `pathlib` 模块中导入 `Path` 类。`Path` 专门用来表示和操作文件路径。

```python
path = Path("data/demo.jsonl")
```

创建一个路径对象 `path`，它表示：

```
data/demo.jsonl
```

也就是当前程序所在目录下：

```
data 文件夹
└── demo.jsonl 文件
```

此时还**没有读取文件**，只是把路径保存到了 `path` 变量中。

```python
with path.open("r", encoding="utf-8") as f:
```

等价于：

```python
with open("data/demo.jsonl", "r", encoding="utf-8") as f:
```

各部分含义：

- `path.open(...)`：打开 `path` 指向的文件
- `"r"`：只读模式
- `encoding="utf-8"`：按照 UTF-8 编码读取
- `as f`：把打开后得到的**文件对象**交给变量 `f`
- `with`：代码块结束后自动关闭文件

因此这里：

```python
path
```

保存的是**文件路径对象**，而：

```python
f
```

保存的是**已经打开的文件对象**。

可以接着逐行读取：

```python
from pathlib import Path

path = Path("data/demo.jsonl")

with path.open("r", encoding="utf-8") as f:
    for line in f:
        print(line)
```

这里的 `demo.jsonl` 不是普通 JSON，而是 **JSON Lines 文件**。通常每一行都是一个独立的 JSON 对象，例如：

```json
{"name": "小明", "age": 18}
{"name": "小红", "age": 19}
{"name": "小刚", "age": 20}
```

读取时一般写成：

```python
import json
from pathlib import Path

path = Path("data/demo.jsonl")

with path.open("r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        print(data["name"])
```

注意这里使用的是：

```python
json.loads(line)
```

因为 `line` 已经是一个字符串。

而之前的：

```python
json.load(f)
```

是直接从整个文件对象 `f` 中读取一个完整 JSON。

#### A.4 第二版再加 JSON 解析和坏行处理

```python
import json
from pathlib import Path

path = Path("data/demo.jsonl")

total_lines = 0
bad_lines = 0
records = []

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
            print(f"坏行 line={line_no}: {line[:50]}")
            continue

        records.append(obj)

print("total_lines =", total_lines)
print("bad_lines =", bad_lines)
print("valid_records =", len(records))
```

这里的关键不是 `json.loads` 这一句，而是：

> 不可信的数据一定要用 `try/except` 包起来。真实数据经常有坏行，程序不能因为一行坏数据直接崩掉。
> 这段代码是在**逐行读取文件，同时给每一行编号**：

```python
for line_no, line in enumerate(f, start=1):
    line = line.strip()
    print(line_no, repr(line))
```

##### A.4 补充：第一行

```python
for line_no, line in enumerate(f, start=1):
```

前面已经打开了文件：

```python
with path.open("r", encoding="utf-8") as f:
```

文件对象 `f` 可以被 `for` 循环遍历，每循环一次，就从文件中读取一行。

不过这里没有直接写：

```python
for line in f:
```

而是写：

```python
enumerate(f, start=1)
```

`enumerate()` 会在遍历内容的同时，给每个内容添加一个编号。

假设文件里有：

```json
{"name": "小明"}
{"name": "小红"}
{"name": "小刚"}
```

那么：

```python
enumerate(f, start=1)
```

会依次产生类似这样的内容：

```python
(1, '{"name": "小明"}\n')
(2, '{"name": "小红"}\n')
(3, '{"name": "小刚"}\n')
```

因此：

```python
line_no, line
```

是在把每个二元组拆开：

- `line_no`：当前行号
- `line`：当前这一行的文本

`start=1` 表示从 `1` 开始编号。否则默认从 `0` 开始：

```python
enumerate(f)           # 0、1、2……
enumerate(f, start=1)  # 1、2、3……
```

---

##### A.4 补充：第二行

```python
line = line.strip()
```

`line` 是一个字符串。文件里的每一行末尾通常带有换行符 `\n`。

例如，读取到的实际字符串可能是：

```python
'{"name": "小明"}\n'
```

调用：

```python
line.strip()
```

会删除字符串首尾的空白字符，例如：

- 空格
- `\n` 换行符
- `\t` 制表符

处理后变成：

```python
'{"name": "小明"}'
```

然后又赋值给 `line`：

```python
line = line.strip()
```

意思是：用处理后的字符串替换原来的 `line`。

---

##### A.4 补充：第三行

```python
print(line_no, repr(line))
```

它会打印：

1. 行号 `line_no`
2. 当前行的字符串表示 `repr(line)`

`repr()` 会把字符串的边界和特殊字符清楚地显示出来。

例如：

```python
line = "hello"
print(line)
```

输出：

```
hello
```

而：

```python
print(repr(line))
```

输出：

```
'hello'
```

可以看到引号，因此更容易判断它是字符串。

再例如：

```python
line = "hello\n"
```

普通打印：

```python
print(line)
```

`\n` 会真的产生换行。

而：

```python
print(repr(line))
```

会显示：

```
'hello\n'
```

所以 `repr()` 很适合用来**检查字符串中有没有换行、空格等不可见字符**。

---

##### A.4 补充：完整运行过程

假设 `demo.jsonl` 内容是：

```json
{"name": "小明"}
{"name": "小红"}

{"name": "小刚"}
```

运行：

```python
from pathlib import Path

path = Path("data/demo.jsonl")

with path.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        print(line_no, repr(line))
```

输出大致是：

```
1 '{"name": "小明"}'
2 '{"name": "小红"}'
3 ''
4 '{"name": "小刚"}'
```

第三行原本是空行，经过 `strip()` 后变成空字符串：

```python
''
```

因此，整段代码可以概括为：

> 打开文件后逐行读取，从第 1 行开始编号，删除每行首尾的空白，并把行号和该行的准确字符串内容打印出来。

#### A.5 第三版再提取文本字段

不同 JSONL 数据集字段名不一定一样，所以你可以写一个小函数：

```python
def get_text(obj: dict) -> str:
    if "text" in obj:
        return str(obj["text"])
    if "instruction" in obj or "output" in obj:
        return (str(obj.get("instruction", "")) + " " + str(obj.get("output", ""))).strip()
    return ""
```

这个函数的思路是：

- 优先用 `text`；
- 如果没有 `text`，就拼接 `instruction` 和 `output`；
- 如果都没有，就返回空字符串。

#### A.6 第四版统计长度、空样本、重复样本和字符频率

可以使用 Python 标准库，不用 pandas：

```python
from collections import Counter

length_buckets = Counter()
char_counter = Counter()
seen_texts = set()
duplicate_count = 0
empty_count = 0

def bucket_length(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 10:
        return "1-10"
    if n <= 50:
        return "11-50"
    if n <= 100:
        return "51-100"
    return "100+"

for obj in records:
    text = get_text(obj).strip()

    if not text:
        empty_count += 1
        continue

    if text in seen_texts:
        duplicate_count += 1
    else:
        seen_texts.add(text)

    length_buckets[bucket_length(len(text))] += 1
    char_counter.update(text)

print("empty_count =", empty_count)
print("duplicate_count =", duplicate_count)
print("length_buckets =", dict(length_buckets))
print("top_chars =", char_counter.most_common(20))
```

这段代码的核心思路：

- `Counter` 适合做“某个东西出现了多少次”；
- `set` 适合做去重和判断是否见过；
- `len(text)` 是文本长度；
- `char_counter.update(text)` 会把字符串当成字符序列，逐字计数。

#### A.7 A 作业自检清单

完成后你至少要能回答：

- 如果某一行不是合法 JSON，程序会不会崩？
- 空文本是怎么判断的？
- 重复文本是按原始字符串判断，还是按 `strip()` 后的字符串判断？
- 高频字符里是否包含空格、标点？如果包含，是否符合你的预期？
- 为什么这个作业要求不能用 pandas？因为它想训练你理解最底层的数据读取和循环处理。

### 练习 B：NumPy 形状体操

给定 `x:[B,T,C]`：

1. 计算每个 token 在 C 维的均值。
2. 把最后两个维度展平为 `[B,T*C]`。
3. 与 `W:[C,V]` 相乘得到 `[B,T,V]`。
4. 对 V 维做稳定 softmax，验证概率和接近 1。

#### B.1 先设定一个能肉眼检查的小形状

不要一开始用很大的数字。先用：

```python
B = 2
T = 3
C = 4
V = 5
```

这表示：

- 一次处理 `2` 个样本；
- 每个样本有 `3` 个 token；
- 每个 token 用 `4` 维向量表示；
- 词表大小或输出类别数是 `5`。

#### B.2 写代码前先写形状注释

这是深度学习代码里非常重要的习惯：

```python
# x: [B, T, C]
# W: [C, V]
# logits: [B, T, V]
```

你应该先在纸上或注释里判断形状，再写代码。否则 NumPy 可能通过广播“帮你算出来”，但结果不是你真正想要的。

#### B.3 代码骨架

新建 `exercise_b_numpy_shapes.py`：

```python
import numpy as np

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
```

先运行，确认输出：

```text
x.shape = (2, 3, 4)
W.shape = (4, 5)
```

#### B.4 第 1 问：计算每个 token 在 C 维的均值

`x` 的形状是 `[B,T,C]`。每个 token 是最后一维 `C` 上的一个向量，所以要沿最后一维求平均：

```python
token_mean = x.mean(axis=-1)
print("token_mean.shape =", token_mean.shape)
```

结果应该是：

```text
token_mean.shape = (2, 3)
```

原因是最后的 `C` 维被平均掉了：

```text
[B, T, C] -> [B, T]
```

#### B.5 第 2 问：把最后两个维度展平为 `[B,T*C]`

```python
flat = x.reshape(B, T * C)
print("flat.shape =", flat.shape)
```

结果应该是：

```text
flat.shape = (2, 12)
```

注意：这里不是把全部维度压成一维，而是保留 batch 维：

```text
[B, T, C] -> [B, T*C]
```

#### B.6 第 3 问：与 `W:[C,V]` 相乘得到 `[B,T,V]`

`x` 是三维，`W` 是二维。NumPy 的 `@` 会把最后一维拿去做矩阵乘法：

```python
logits = x @ W
print("logits.shape =", logits.shape)
```

形状变化是：

```text
[B, T, C] @ [C, V] -> [B, T, V]
```

也就是：

```text
(2, 3, 4) @ (4, 5) -> (2, 3, 5)
```

#### B.7 第 4 问：稳定 softmax，并验证概率和

先写稳定版 softmax：

```python
def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)
```

然后对最后一维 `V` 做 softmax：

```python
probs = softmax(logits, axis=-1)
print("probs.shape =", probs.shape)
```

验证每个位置的概率和是否接近 1：

```python
prob_sums = probs.sum(axis=-1)
print("prob_sums.shape =", prob_sums.shape)
print(prob_sums)
print("all close to 1:", np.allclose(prob_sums, 1.0))
```

`prob_sums.shape` 应该是：

```text
(2, 3)
```

因为你把最后一维 `V` 加掉了：

```text
[B, T, V] -> [B, T]
```

#### B.8 B 作业自检清单

完成后你至少要能解释：

- 为什么 `x.mean(axis=-1)` 的结果是 `[B,T]`？
- 为什么 `x.reshape(B, T*C)` 要保留 `B`？
- 为什么 `[B,T,C] @ [C,V]` 可以得到 `[B,T,V]`？
- 为什么 softmax 要在 `axis=-1` 上做？
- 为什么验证概率和时，要看 `probs.sum(axis=-1)`？

### 练习 C：线性回归

仅用 NumPy 生成 `y=3x+2+noise`，手写均方误差、梯度下降和训练曲线；观察学习率 `0.0001、0.01、1.0` 的差异。

#### C.1 先理解你要训练什么

我们假设真实数据来自：

$$
y = 3x + 2 + \text{noise}
$$

你的模型不知道真实参数是 `3` 和 `2`，它只能从数据里学出：

$$
\hat{y} = wx + b
$$

训练目标是让预测值 $\hat{y}$ 尽量接近真实值 $y$。

均方误差是：

$$
L = \frac{1}{n}\sum_i(\hat{y}_i-y_i)^2
$$

梯度是：

$$
\frac{\partial L}{\partial w}
= \frac{2}{n}\sum_i(\hat{y}_i-y_i)x_i
$$

$$
\frac{\partial L}{\partial b}
= \frac{2}{n}\sum_i(\hat{y}_i-y_i)
$$

参数更新是：

$$
w \leftarrow w - \eta \frac{\partial L}{\partial w}
$$

$$
b \leftarrow b - \eta \frac{\partial L}{\partial b}
$$

其中 $\eta$ 就是学习率。

#### C.2 第一版先生成数据

新建 `exercise_c_linear_regression.py`：

```python
import numpy as np

rng = np.random.default_rng(42)

n = 100
x = rng.uniform(-5, 5, size=n)
noise = rng.normal(0, 1.0, size=n)
y = 3 * x + 2 + noise

print(x[:5])
print(y[:5])
```

先确认你真的得到了数据。此时还没有训练。

```python
rng = np.random.default_rng(42)
```

这句是在创建一个 NumPy 的**随机数生成器**，并把它保存到变量 `rng` 中。

前提通常是：

```python
import numpy as np
```

逐部分看：

```python
np.random.default_rng(42)
```

- `np`：NumPy 模块
- `random`：NumPy 中负责随机数的部分
- `default_rng()`：创建一个新的随机数生成器
- `42`：随机种子，也叫 seed

`rng` 是变量名，通常是 random number generator 的缩写。

创建后可以这样生成随机数：

```python
rng.random()
```

例如可能得到：

```
0.7739560485559633
```

生成 5 个随机数：

```python
rng.random(5)
```

生成 1 到 9 之间的随机整数：

```python
rng.integers(1, 10)
```

注意上限 `10` 不包含在内，因此可能生成的是：

```
1, 2, 3, ..., 9
```

##### C.2 补充：为什么要写 `42`

`42` 的作用是让随机结果**可以重复出现**。

例如每次重新运行：

```python
rng = np.random.default_rng(42)
print(rng.integers(1, 10, size=5))
```

都会得到同一组结果：

```
[1 7 6 4 4]
```

这里并不是真的“不随机”，而是使用相同的起点，按照固定算法生成同一串伪随机数。

种子换了，结果通常也会变：

```python
rng = np.random.default_rng(100)
```

不写种子：

```python
rng = np.random.default_rng()
```

那么每次运行通常会得到不同结果。

##### C.2 补充：一个容易忽略的地方

同一个 `rng` 每调用一次，都会继续向后生成新的数字：

```python
rng = np.random.default_rng(42)

print(rng.integers(1, 10, size=3))
print(rng.integers(1, 10, size=3))
```

两次输出通常不同，因为生成器的内部状态已经前进了。

但重新创建：

```python
rng = np.random.default_rng(42)
```

就会重新回到相同的起点。

所以这句话可以概括为：

> 创建一个随机数生成器，并用种子 42 固定它的初始状态，让程序每次运行时能够复现相同的随机结果。

#### C.3 第二版写预测函数和 loss

```python
def predict(x, w, b):
    return w * x + b

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

w = 0.0
b = 0.0

y_pred = predict(x, w, b)
loss = mse_loss(y_pred, y)

print("initial loss =", loss)
```

这里的重点是：先让前向计算跑起来。不要一开始就写训练循环。

```python
token_mean = x.mean(axis=-1)
```

意思是：**对数组 `x` 的最后一个维度求平均值**，并把结果保存到 `token_mean`。

##### C.3 补充：`x.mean()`

`mean()` 是求平均值的方法。

例如：

```python
x = np.array([1, 2, 3])
x.mean()
```

结果是：

```
2.0
```

##### C.3 补充：`axis=-1`

`axis` 指定“沿哪个维度进行计算”。

`-1` 表示**最后一个维度**。因此：

```python
x.mean(axis=-1)
```

等价于：

```python
x.mean(axis=x.ndim - 1)
```

例如 `x` 是二维数组：

```python
x = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
```

它的形状是：

```python
x.shape
# (2, 3)
```

最后一个维度的长度是 `3`，因此会对每一行的三个数求平均：

```python
token_mean = x.mean(axis=-1)
```

得到：

```
[2. 5.]
```

计算过程是：

```
第一行：(1 + 2 + 3) / 3 = 2
第二行：(4 + 5 + 6) / 3 = 5
```

原来的形状：

```
(2, 3)
```

求平均后的形状：

```
(2,)
```

最后一个维度被平均掉了。

---

在神经网络中，`x` 可能是三维数组：

```
(batch_size, token数量, 特征维度)
```

例如：

```
(4, 10, 768)
```

执行：

```python
token_mean = x.mean(axis=-1)
```

就是对每个 token 的 `768` 个特征取平均：

```
(4, 10, 768) → (4, 10)
```

所以变量叫 `token_mean`，很可能表示：

> 每个 token 在其全部特征维度上的平均值。

需要注意，`axis=-1` 并不固定表示“列”，它始终表示**最后一个维度**。对于二维数组，它看起来像按行求平均；对于三维或更高维数组，则是对最里面那一层求平均。

#### C.4 第三版手写梯度

```python
def gradients(x, y_true, y_pred):
    n = len(x)
    error = y_pred - y_true
    dw = (2 / n) * np.sum(error * x)
    db = (2 / n) * np.sum(error)
    return dw, db
```

你可以先打印一次梯度：

```python
dw, db = gradients(x, y, y_pred)
print("dw =", dw)
print("db =", db)
```

这一步的意义是确认：你的程序已经知道应该往哪个方向调整 `w` 和 `b`。

#### C.5 第四版写训练循环

```python
def train(lr, steps=200):
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
            print(f"lr={lr} step={step} loss={loss:.4f} w={w:.4f} b={b:.4f}")

    return w, b, losses
```

然后分别测试三个学习率：

```python
for lr in [0.0001, 0.01, 1.0]:
    print("=" * 60)
    w, b, losses = train(lr)
    print(f"final lr={lr}: w={w:.4f}, b={b:.4f}, final_loss={losses[-1]:.4f}")
```

可以。你先把 `print()` 理解成：

> **让 Python 把括号里面的内容显示到屏幕上。**

##### C.5 print 补充 1：最简单的 `print`

```python
print("hello")
```

输出：

```
hello
```

这里：

- `print` 是 Python 内置的打印函数
- `()` 里面放你想显示的东西
- `"hello"` 是字符串，也就是一段文字

再比如：

```python
print(123)
```

输出：

```
123
```

数字不需要引号；文字需要引号。

```python
print("123")  # 字符串
print(123)    # 整数
```

虽然显示起来相似，但类型不同。

---

##### C.5 print 补充 2：打印变量

```python
lr = 0.01
print(lr)
```

Python 会先找到变量 `lr` 保存的值，再显示它：

```
0.01
```

重点是：

```python
print(lr)
```

打印的是变量的值。

而：

```python
print("lr")
```

打印的是文字 `lr`：

```
lr
```

两者完全不同。

---

##### C.5 print 补充 3：一次打印多个内容

```python
lr = 0.01
step = 20

print(lr, step)
```

输出：

```
0.01 20
```

`print()` 中可以放多个内容，用逗号隔开：

```python
print("lr =", lr, "step =", step)
```

输出：

```
lr = 0.01 step = 20
```

Python 默认会在这些内容之间加入一个空格。

因此，你之前那句话其实可以先写成：

```python
print("lr=", lr, "step=", step, "loss=", loss)
```

只是这样输出时可能会出现多余空格：

```
lr= 0.01 step= 20 loss= 0.123456
```

---

##### C.5 print 补充 4：什么是 f-string

f-string 是一种更方便的字符串写法。

```python
lr = 0.01
print(f"lr={lr}")
```

输出：

```
lr=0.01
```

先看外层：

```python
f"lr={lr}"
```

它整体仍然是一个字符串。

字符串前面的 `f` 告诉 Python：

> 这个字符串里面的 `{}` 不是普通文字，请计算其中的内容，并把结果放进字符串。

所以：

```python
f"lr={lr}"
```

会先变成：

```python
"lr=0.01"
```

然后 `print()` 再把它显示出来。

整个过程可以理解为：

```
变量 lr 的值是 0.01
        ↓
f"lr={lr}"
        ↓
"lr=0.01"
        ↓
print(...)
        ↓
屏幕显示 lr=0.01
```

---

##### C.5 print 补充 5：`{}` 里面可以放什么

可以放变量：

```python
name = "Polar"
print(f"name={name}")
```

输出：

```
name=Polar
```

也可以放计算表达式：

```python
a = 3
b = 4

print(f"a+b={a+b}")
```

输出：

```
a+b=7
```

注意：

```python
"a+b="
```

是普通文字，而：

```python
{a+b}
```

会真的执行加法。

---

##### C.5 print 补充 6：回到你的代码

```python
print(f"lr={lr} step={step} loss={loss:.4f} w={w:.4f} b={b:.4f}")
```

这里 `print()` 的括号里实际上只有**一个东西**：

```python
f"lr={lr} step={step} loss={loss:.4f} w={w:.4f} b={b:.4f}"
```

这是一个 f-string。

假设：

```python
lr = 0.01
step = 20
loss = 0.123456
w = 1.987654
b = -0.456789
```

Python 会逐个替换：

```python
{lr}        → 0.01
{step}      → 20
{loss:.4f}  → 0.1235
{w:.4f}     → 1.9877
{b:.4f}     → -0.4568
```

于是整个字符串变成：

```
lr=0.01 step=20 loss=0.1235 w=1.9877 b=-0.4568
```

最后 `print()` 把它显示出来。

---

##### C.5 print 补充 7：`:.4f` 到底是什么

以：

```python
{loss:.4f}
```

为例。

可以拆成：

```
{ loss : .4f }
  变量   显示格式
```

- `loss`：取变量 `loss` 的值
- `:`：后面开始指定显示格式
- `.4f`：用小数形式显示，保留四位小数

例如：

```python
x = 3.1415926

print(f"{x}")
print(f"{x:.2f}")
print(f"{x:.4f}")
```

输出：

```
3.1415926
3.14
3.1416
```

这里并没有修改变量本身：

```python
print(x)
```

仍然会输出：

```
3.1415926
```

`:.4f` 只是规定这一次打印时怎么显示。

---

##### C.5 print 补充 8：为什么外面要有文字，里面还要有变量

例如：

```python
print(f"loss={loss:.4f}")
```

其中：

```
loss=
```

只是给人看的标签。

而：

```python
{loss:.4f}
```

才是变量的实际值。

假设直接写：

```python
print(loss)
```

输出：

```
0.1235
```

你只能看到一个数字，不一定知道它代表什么。

写成：

```python
print(f"loss={loss:.4f}")
```

输出：

```
loss=0.1235
```

就能看出这个数字是损失值。

---

##### C.5 print 补充 9：`print()` 默认会换行

```python
print("A")
print("B")
```

输出：

```
A
B
```

因为每次 `print()` 结束后默认添加换行。

不想换行，可以写：

```python
print("A", end="")
print("B")
```

输出：

```
AB
```

或者：

```python
print("A", end=" ")
print("B")
```

输出：

```
A B
```

---

你目前可以先记住这个核心结构：

```python
变量 = 某个值
print(变量)              # 打印变量值
print("普通文字")         # 打印固定文字
print(f"文字{变量}")      # 把变量值插进文字
print(f"{变量:.4f}")      # 把数字保留四位小数后显示
```

#### C.6 你应该观察到什么

学习率 `0.0001`：

- 通常很稳定；
- 但下降很慢；
- `w` 和 `b` 可能离 `3` 和 `2` 还比较远。

学习率 `0.01`：

- 通常比较合适；
- loss 会明显下降；
- `w` 会接近 `3`，`b` 会接近 `2`。

学习率 `1.0`：

- 很可能震荡或发散；
- loss 可能越来越大；
- 参数可能变成非常大的数。

你要写进报告里的不是“哪个数字最好看”，而是：

> 学习率太小会学得慢；合适的学习率能稳定收敛；学习率太大可能越过最低点，导致震荡甚至发散。

#### C.7 如果要画训练曲线

如果环境里有 `matplotlib`，可以加：

```python
import matplotlib.pyplot as plt

for lr in [0.0001, 0.01, 1.0]:
    _, _, losses = train(lr)
    plt.plot(losses, label=f"lr={lr}")

plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.savefig("linear_regression_losses.png", dpi=150)
```

如果暂时不会画图，也可以先不画，直接打印每 20 步的 loss。作业重点是你能解释训练过程。
这段代码的作用是：

> 用三种不同的学习率分别训练模型，把每次训练过程中的 `loss` 画在同一张图上，最后保存成图片。

```python
import matplotlib.pyplot as plt

for lr in [0.0001, 0.01, 1.0]:
    _, _, losses = train(lr)
    plt.plot(losses, label=f"lr={lr}")

plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.savefig("linear_regression_losses.png", dpi=150)
```

##### C.7 画图补充 1：导入画图工具

```python
import matplotlib.pyplot as plt
```

- `matplotlib`：Python 常用绘图库
- `pyplot`：其中负责画图的模块
- `as plt`：把它简称为 `plt`

之后就可以写：

```python
plt.plot(...)
plt.xlabel(...)
```

而不用每次写完整名称。

---

##### C.7 画图补充 2：依次测试三种学习率

```python
for lr in [0.0001, 0.01, 1.0]:
```

这个循环会执行三次：

```python
lr = 0.0001
lr = 0.01
lr = 1.0
```

也就是说，用三种学习率各训练一次模型。

---

##### C.7 画图补充 3：调用训练函数

```python
_, _, losses = train(lr)
```

假设 `train(lr)` 返回三个结果，例如：

```python
return w, b, losses
```

那么本来可以写：

```python
w, b, losses = train(lr)
```

但这里不关心训练后的 `w` 和 `b`，只关心每一步的损失 `losses`，所以写成：

```python
_, _, losses = train(lr)
```

下划线 `_` 通常表示：

> 这个返回值我暂时不需要。

例如：

```python
result = train(0.01)
```

假设得到：

```python
(2.01, 0.98, [10.2, 7.5, 4.1, 2.0])
```

那么：

```python
_, _, losses = result
```

相当于：

```python
_ = 2.01
_ = 0.98
losses = [10.2, 7.5, 4.1, 2.0]
```

最终真正使用的是 `losses`。

需要注意，两个 `_` 实际上是同一个普通变量，只是大家约定用它表示“不关心”。这里没有问题，因为这些值之后没有被使用。

---

##### C.7 画图补充 4：把损失画成曲线

```python
plt.plot(losses, label=f"lr={lr}")
```

假设：

```python
losses = [10, 7, 4, 2, 1]
```

`plt.plot(losses)` 会默认把列表的位置当作横坐标：

```
横坐标：0  1  2  3  4
纵坐标：10 7  4  2  1
```

因此它画出的点相当于：

```
(0, 10)
(1, 7)
(2, 4)
(3, 2)
(4, 1)
```

再把这些点连接起来。

这里的横坐标就代表训练步数 `step`，纵坐标代表每一步的 `loss`。

##### C.7 画图补充：`label=f"lr={lr}"`

这是给当前曲线起名字。

第一次循环：

```python
lr = 0.0001
```

于是：

```python
label=f"lr={lr}"
```

变成：

```python
label="lr=0.0001"
```

第二、三次分别变成：

```python
label="lr=0.01"
label="lr=1.0"
```

这些名字之后会显示在图例中。

三次循环并不是生成三张图，而是在**当前同一张图上连续添加三条线**。

---

##### C.7 画图补充 5：设置横轴名称

```python
plt.xlabel("step")
```

让横轴下方显示：

```
step
```

表示训练步骤。

##### C.7 画图补充 6：设置纵轴名称

```python
plt.ylabel("loss")
```

让纵轴旁边显示：

```
loss
```

表示损失值。

---

##### C.7 画图补充 7：显示图例

```python
plt.legend()
```

图例用来说明每条曲线对应哪个学习率。

因为前面为每条曲线设置了：

```python
label=f"lr={lr}"
```

所以图例中会显示：

```
lr=0.0001
lr=0.01
lr=1.0
```

没有 `plt.legend()`，这些 `label` 不会自动显示出来。

---

##### C.7 画图补充 8：保存图片

```python
plt.savefig("linear_regression_losses.png", dpi=150)
```

把当前绘制的图保存成：

```
linear_regression_losses.png
```

通常会保存在程序运行时的**当前工作目录**中。

##### C.7 画图补充：`dpi=150`

`dpi` 表示图片分辨率，可以粗略理解为保存图片的清晰程度。

- `dpi=72`：较低
- `dpi=100`：普通
- `dpi=150`：比较清晰
- `dpi=300`：适合论文或打印，但文件更大

它不会影响训练结果，只影响保存图片的清晰度。

---

##### C.7 画图补充：整个循环实际相当于

```python
_, _, losses = train(0.0001)
plt.plot(losses, label="lr=0.0001")

_, _, losses = train(0.01)
plt.plot(losses, label="lr=0.01")

_, _, losses = train(1.0)
plt.plot(losses, label="lr=1.0")
```

然后统一添加坐标名称、图例并保存：

```python
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.savefig("linear_regression_losses.png", dpi=150)
```

最后得到的图一般用于比较：

- 学习率太小时，损失下降很慢；
- 学习率合适时，损失较快且稳定下降；
- 学习率太大时，损失可能震荡甚至越来越大。

另外，这段代码只保存图片，不一定弹出窗口。想在运行时显示图片，可以最后加：

```python
plt.show()
```

#### C.8 C 作业自检清单

完成后你至少要能回答：

- 为什么模型形式写成 $\hat{y}=wx+b$？
- loss 为什么用平方误差？
- `dw` 和 `db` 分别表示什么？
- 学习率太小、合适、太大分别会发生什么？
- 最终学到的 `w` 和 `b` 为什么不会刚好等于 `3` 和 `2`？因为数据里有噪声。

### 作业报告怎么写

每个练习写一个很短的报告即可，不要写成论文。可以放在 `notes.md` 里：

```markdown
## Exercise A

### 我做了什么

读取 JSONL，统计空样本、重复样本、长度分桶和高频字符。

### 我如何验证

先用 6 行 demo 数据测试，其中包含坏行、空文本和重复文本。

### 结果

- 总行数：
- 坏行数：
- 空样本数：
- 重复样本数：
- 高频字符：

### 我遇到的问题

例如：文件路径一开始写错；坏行需要 try/except；空字符串要先 strip。

### 我的结论

纯 Python 也能完成基础文本清洗；真实数据必须考虑坏行、空字段和重复样本。
```

练习 B 和 C 也按这个模板写。你只要能把“做了什么、怎么验证、结果是什么、我学到什么”讲清楚，就已经达到了这个模块的目的。

## 验收与复盘

- 能解释 list 与 ndarray、参数与超参数、训练/验证/测试的区别。
- 看到 `[B,T,C] @ [C,V]` 能立即说出输出形状。
- 能手写稳定 Softmax、交叉熵和数值梯度检查。
- 能读写 UTF-8 JSONL，并处理坏行与空字段。
- 能用 Git 保存练习，每个实验有假设和结论。

## 课后 Project

先完成上面的讲解和验收，再独立完成下面三个 Project。完整参考答案见：[模块 1 综合练习标准答案](01_Python_Git_数学基础_综合练习标准答案.md)。

### Project 1：纯 Python 文本分析

新建 `module01_homework/exercise_a_jsonl_stats.py`，只使用 Python 标准库完成 JSONL 文本统计，不能使用 pandas。

要求：

1. 读取一个 UTF-8 编码的 JSONL 文件。
2. 跳过空行。
3. 对非法 JSON 行做容错处理：记录坏行数量，但程序不能崩溃。
4. 从每条记录中提取文本：
   - 优先使用 `text` 字段；
   - 如果没有 `text`，则拼接 `instruction` 和 `output`；
   - 如果都没有，则视为空文本。
5. 统计总行数、合法 JSON 记录数、坏行数、空文本数、重复文本数、文本长度分桶与高频字符 Top 20。
6. 写一段简短结论：你认为这个数据集有什么明显问题？

### Project 2：NumPy 形状体操

新建 `module01_homework/exercise_b_numpy_shapes.py`，使用 NumPy 完成张量形状练习。

给定：

```python
B = 2
T = 3
C = 4
V = 5
x.shape == (B, T, C)
W.shape == (C, V)
```

要求：

1. 计算每个 token 在 `C` 维上的均值，输出形状应为 `[B,T]`。
2. 把 `x` 的最后两个维度展平为 `[B,T*C]`。
3. 计算 `logits = x @ W`，输出形状应为 `[B,T,V]`。
4. 对 `logits` 的最后一维做稳定 softmax。
5. 验证每个 `[B,T]` 位置上的概率和都接近 `1`。
6. 在注释或报告里解释每一步形状为什么这样变化。

### Project 3：NumPy 线性回归

新建 `module01_homework/exercise_c_linear_regression.py`，只用 NumPy 训练一个一元线性回归模型。

要求：

1. 随机生成数据：

   $$
   y = 3x + 2 + \text{noise}
   $$

2. 模型形式为：

   $$
   \hat{y} = wx + b
   $$

3. 手写均方误差 MSE、`w` 和 `b` 的梯度，并使用梯度下降训练。
4. 分别测试学习率 `0.0001`、`0.01`、`1.0`，打印 loss、`w`、`b` 的变化。
5. 保存 loss 曲线（可选）并解释：学习率太小、合适、太大会分别发生什么。

### 统一提交物

```text
module01_homework/
├── exercise_a_jsonl_stats.py
├── exercise_b_numpy_shapes.py
├── exercise_c_linear_regression.py
├── data/
│   └── demo.jsonl
├── linear_regression_losses.png   # 可选
└── notes.md
```
