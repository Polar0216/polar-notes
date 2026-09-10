# Conda 使用指南

你之前用过 Python 的 `venv`。可以先把 Conda 理解成：

> **Conda = 虚拟环境管理器 + 软件包管理器。**

它不仅能隔离 Python 包，还能管理不同版本的 Python，以及 CUDA、C/C++ 库等非 Python 依赖。每个 Conda 环境都拥有自己独立的 Python、软件包和依赖，不会轻易互相干扰。

---

## 一、Conda、Anaconda、Miniconda 是什么关系

### Conda

`conda` 是命令行工具，负责：

- 创建和删除环境
- 安装和卸载软件包
- 管理 Python 版本
- 解决软件包依赖关系

### Anaconda

Anaconda 是一个很大的发行版，安装后自带：

- Conda
- Python
- NumPy
- pandas
- Jupyter
- 大量数据科学软件包

优点是开箱即用，缺点是占用空间较大。

### Miniconda

Miniconda 只预装：

- Conda
- Python
- 少量基础组件

之后需要什么再安装什么。

对于你这种会自己配置开发环境、又在意磁盘空间的情况，通常 **Miniconda 更合适**。

---

# 二、安装后先检查 Conda

Windows 上可以先打开：

- Anaconda Prompt
- Miniconda Prompt
- PowerShell

输入：

```powershell
conda --version
```

例如输出：

```
conda 26.5.2
```

说明 Conda 已经安装。

查看 Conda 的详细信息：

```powershell
conda info
```

查看帮助：

```powershell
conda --help
```

查看某条命令的帮助：

```powershell
conda create --help
```

Conda 官方也建议通过 `--help` 查看每条命令的完整参数。

---

# 三、最重要的概念：Conda 环境

假设你同时做三个项目：

```
项目 A：Python 3.10 + PyTorch
项目 B：Python 3.12 + pandas
项目 C：Python 3.11 + TensorFlow
```

如果把它们都装进同一个 Python，可能出现版本冲突。

Conda 可以分别创建三个独立环境：

```
base
├── pytorch_env
├── data_env
└── tensorflow_env
```

每个环境就像一个独立的小型 Python 工作区。

---

# 四、查看已有环境

```powershell
conda env list
```

或者：

```powershell
conda info --envs
```

可能输出：

```
# conda environments:
#
base                  *  C:\Users\Polar\miniconda3
llm                      C:\Users\Polar\miniconda3\envs\llm
pytorch                  C:\Users\Polar\miniconda3\envs\pytorch
```

其中：

```
*
```

表示当前正在使用的环境。

这里的 `base` 是 Conda 自己的基础环境。

---

# 五、创建环境

## 1. 创建一个只有基础组件的环境

```powershell
conda create --name myenv
```

简写：

```powershell
conda create -n myenv
```

其中：

```
conda create    创建环境
-n              --name 的缩写
myenv           环境名称
```

---

## 2. 创建环境并指定 Python 版本

推荐这样创建：

```powershell
conda create -n myenv python=3.12
```

意思是：

> 创建一个名为 `myenv` 的环境，并在里面安装 Python 3.12。

Conda 支持在不同环境中安装不同版本的 Python。

---

## 3. 创建环境时同时安装软件包

```powershell
conda create -n data python=3.12 numpy pandas matplotlib
```

这样会一次性创建环境并安装：

```
Python 3.12
NumPy
pandas
Matplotlib
```

Conda 官方的 `conda create` 命令支持在创建环境时直接指定需要安装的软件包。

---

# 六、激活环境

创建后不会自动进入环境，需要激活：

```powershell
conda activate myenv
```

激活成功后，命令行前面通常会出现：

```
(myenv) PS C:\Users\Polar>
```

这表示：

> 现在执行的 `python`、`pip` 和 `conda install`，默认都针对 `myenv` 环境。

检查当前使用的 Python：

```powershell
where python
```

或者：

```powershell
python --version
```

还可以在 Python 中查看解释器路径：

```powershell
python -c "import sys; print(sys.executable)"
```

输出可能是：

```
C:\Users\Polar\miniconda3\envs\myenv\python.exe
```

`conda activate` 本质上会修改当前终端的环境变量，使当前环境中的程序优先被找到。

---

# 七、退出环境

```powershell
conda deactivate
```

例如：

```
(myenv) PS C:\Users\Polar>
```

执行：

```powershell
conda deactivate
```

可能回到：

```
(base) PS C:\Users\Polar>
```

再执行一次：

```powershell
conda deactivate
```

可能完全退出 Conda 环境：

```
PS C:\Users\Polar>
```

---

# 八、安装软件包

先进入对应环境：

```powershell
conda activate myenv
```

然后安装：

```powershell
conda install numpy
```

安装多个：

```powershell
conda install numpy pandas matplotlib
```

指定版本：

```powershell
conda install numpy=2.0
```

Conda 会尝试寻找一组彼此兼容的软件包版本，而不是只机械地安装某一个文件。

---

# 九、查看已安装的软件包

```powershell
conda list
```

输出类似：

```
# packages in environment at C:\Users\Polar\miniconda3\envs\myenv:
#
numpy       2.0.1
pip         25.1
python      3.12.4
setuptools  78.1.1
```

查看某个包：

```powershell
conda list numpy
```

---

# 十、搜索软件包

```powershell
conda search pytorch
```

不过安装 PyTorch、CUDA 等软件时，最好按照项目官方网站给出的命令安装，因为不同显卡和 CUDA 版本对应的命令可能不同。

---

# 十一、更新软件包

更新某个软件包：

```powershell
conda update numpy
```

更新 Conda 本身：

```powershell
conda update -n base conda
```

更新当前环境中的全部软件包：

```powershell
conda update --all
```

`conda update --all` 有时会同时升级或降级多个软件包。重要项目不要随便运行，最好先导出环境配置。

---

# 十二、卸载软件包

```powershell
conda remove numpy
```

也可以写：

```powershell
conda uninstall numpy
```

一般使用 `conda remove` 更常见。

---

# 十三、删除整个环境

先退出需要删除的环境：

```powershell
conda deactivate
```

然后：

```powershell
conda remove -n myenv --all
```

例如：

```powershell
conda remove -n pytorch --all
```

删除后检查：

```powershell
conda env list
```

不要直接在文件资源管理器里删除环境文件夹。通过 Conda 删除通常更干净。

---

# 十四、Conda 和 pip 怎么配合

这是最容易混乱的地方。

## 优先原则

在 Conda 环境里，通常采用：

1. 优先用 `conda install`
2. Conda 找不到时再用 `pip install`
3. 先装 Conda 包，最后再装 pip 包

例如：

```powershell
conda activate myenv

conda install numpy pandas
python -m pip install transformers
```

Conda 和 pip 使用的包格式、依赖系统不完全相同，因此混合安装存在一定兼容性限制。

---

## 为什么建议写 `python -m pip`

可以写：

```powershell
pip install requests
```

但更推荐：

```powershell
python -m pip install requests
```

因为这句话明确表示：

> 使用当前这个 Python 解释器对应的 pip。

这样不容易误装到其他环境。

检查位置：

```powershell
where python
where pip
```

它们应当都位于同一个环境，例如：

```
C:\Users\Polar\miniconda3\envs\myenv\python.exe
C:\Users\Polar\miniconda3\envs\myenv\Scripts\pip.exe
```

---

# 十五、Conda 的 channel 是什么

Conda 安装包时，需要从软件仓库下载。这个仓库被称为：

```
channel
```

常见的有：

```
defaults
conda-forge
pytorch
nvidia
```

临时从 `conda-forge` 安装：

```powershell
conda install -c conda-forge package_name
```

例如：

```powershell
conda install -c conda-forge opencv
```

其中：

```
-c conda-forge
```

表示这一次从 `conda-forge` channel 查找软件包。

不要无规律混用很多 channel，否则可能产生依赖来源混乱。通常一个环境尽量以某个主要 channel 为主。

---

# 十六、导出环境

假设你已经配置好一个项目环境：

```powershell
conda activate myenv
```

导出：

```powershell
conda env export > environment.yml
```

会生成类似：

```yaml
name: myenv

dependencies:
  - python=3.12
  - numpy
  - pandas
  - matplotlib
  - pip
  - pip:
      - transformers
```

这个文件可以：

- 备份环境
- 发给别人
- 在其他电脑上恢复
- 放进 Git 项目

Conda 官方支持通过环境定义文件创建、导出和共享环境。

---

# 十七、通过 environment.yml 恢复环境

进入 `environment.yml` 所在文件夹：

```powershell
cd 项目目录
```

执行：

```powershell
conda env create -f environment.yml
```

如果文件中写了：

```yaml
name: myenv
```

就会创建 `myenv` 环境。

也可以指定新名称：

```powershell
conda env create -f environment.yml -n newenv
```

---

# 十八、更新已有环境

当项目中的 `environment.yml` 更新后，可以执行：

```powershell
conda env update -n myenv -f environment.yml
```

如果还希望删除配置文件中已经不存在的软件包：

```powershell
conda env update -n myenv -f environment.yml --prune
```

---

# 十九、复制环境

```powershell
conda create -n newenv --clone oldenv
```

例如：

```powershell
conda create -n pytorch_test --clone pytorch
```

这适合在升级重要软件包前做一个副本。

---

# 二十、清理 Conda 缓存

Conda 下载的软件包和压缩文件会留在缓存中，时间久了可能占用不少空间。

查看将清理什么：

```powershell
conda clean --all --dry-run
```

确认后清理：

```powershell
conda clean --all
```

如果不想逐项确认：

```powershell
conda clean --all -y
```

注意：这主要清理缓存，不会删除你创建的正常环境。

---

# 二十一、关闭自动进入 base 环境

安装 Conda 后，每次打开 PowerShell，可能自动显示：

```
(base)
```

如果不喜欢，可以关闭：

```powershell
conda config --set auto_activate_base false
```

之后重新打开终端。

需要使用环境时手动执行：

```powershell
conda activate myenv
```

重新开启：

```powershell
conda config --set auto_activate_base true
```

---

# 二十二、PowerShell 提示 conda activate 不可用

例如出现：

```
CondaError: Run 'conda init' before 'conda activate'
```

执行：

```powershell
conda init powershell
```

然后：

1. 关闭当前 PowerShell
2. 重新打开 PowerShell
3. 再运行：

```powershell
conda activate myenv
```

因为 `conda activate` 需要修改当前 shell 的状态，所以 Conda 需要先针对 PowerShell 进行初始化。

如果是 CMD：

```cmd
conda init cmd.exe
```

---

# 二十三、在 VS Code 中使用 Conda

## 1. 创建环境

```powershell
conda create -n project python=3.12
conda activate project
```

## 2. 安装项目需要的包

```powershell
conda install numpy matplotlib
```

## 3. 打开 VS Code

```powershell
code .
```

## 4. 选择解释器

在 VS Code 中：

```
Ctrl + Shift + P
```

搜索：

```
Python: Select Interpreter
```

选择类似：

```
Python 3.12 ('project': conda)
```

或者选择路径：

```
C:\Users\Polar\miniconda3\envs\project\python.exe
```

之后 VS Code 运行 Python 时就会使用这个 Conda 环境。

---

# 二十四、Conda 和 venv 的区别

| 项⁠目 | Conda | venv |
| --- | --- | --- |
| 管⁠理 Python 环⁠境 | 可⁠以 | 可⁠以 |
| 管⁠理 Python 版⁠本 | 可⁠以 | 通⁠常⁠不⁠负⁠责 |
| 安⁠装 Python 包 | `conda` 或 `pip` | 主⁠要⁠使⁠用 `pip` |
| 管⁠理 CUDA/C++ 库 | 可⁠以 | 通⁠常⁠不⁠行 |
| 占⁠用⁠空⁠间 | 较⁠大 | 较⁠小 |
| 依⁠赖⁠解⁠析 | 较⁠完⁠整⁠但⁠可⁠能⁠较⁠慢 | 主⁠要⁠由 pip 处⁠理 |
| 适⁠合⁠场⁠景 | AI、科⁠学⁠计⁠算、复⁠杂⁠依⁠赖 | 普⁠通 Python 项⁠目 |

例如，`venv`：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy
```

对应的 Conda 写法：

```powershell
conda create -n project python=3.12
conda activate project
conda install numpy
```

二者的核心目标一样：

> 给项目创建相互隔离的 Python 环境。

区别是 Conda 管得更多。

---

# 二十五、推荐的项目工作流

假设要建立一个机器学习项目：

```powershell
mkdir ml_project
cd ml_project
```

创建环境：

```powershell
conda create -n ml_project python=3.12
```

进入环境：

```powershell
conda activate ml_project
```

安装基础包：

```powershell
conda install numpy pandas matplotlib scikit-learn
```

安装 Conda 没有的包：

```powershell
python -m pip install 包名
```

检查解释器：

```powershell
python -c "import sys; print(sys.executable)"
```

导出环境：

```powershell
conda env export > environment.yml
```

退出：

```powershell
conda deactivate
```

---

# 二十六、你最常用的命令速查

```powershell
# 查看 Conda 版本
conda --version

# 查看所有环境
conda env list

# 创建环境
conda create -n myenv python=3.12

# 激活环境
conda activate myenv

# 退出环境
conda deactivate

# 安装包
conda install numpy

# 从 conda-forge 安装
conda install -c conda-forge opencv

# 查看已安装的软件包
conda list

# 更新包
conda update numpy

# 卸载包
conda remove numpy

# 删除环境
conda remove -n myenv --all

# 导出环境
conda env export > environment.yml

# 通过配置文件创建环境
conda env create -f environment.yml

# 复制环境
conda create -n newenv --clone oldenv

# 清理缓存
conda clean --all

# 初始化 PowerShell
conda init powershell
```

你现阶段只要先熟练下面五条就够了：

```powershell
conda create -n 环境名 python=版本
conda activate 环境名
conda install 包名
conda env list
conda deactivate
```