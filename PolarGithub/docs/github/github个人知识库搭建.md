# Markdown 学习笔记保存与建站总结

## 一、最终目标

你的学习笔记使用普通 `.md` 文件保存，然后通过下面这套工具链管理：

```
Obsidian / VS Code 写 Markdown
              ↓
Git 记录版本
              ↓
GitHub 保存远程仓库
              ↓
GitHub Actions 自动运行 MkDocs
              ↓
MkDocs 生成 HTML
              ↓
GitHub Pages 托管网站
```

以后每次更新笔记，只需要：

```powershell
git add .
git commit -m "更新学习笔记"
git push
```

GitHub 会自动重新生成并更新网站。

---

## 二、GitHub 和 OneDrive 的区别

### GitHub

GitHub 可以理解为：

> 云端的 Git 仓库。

适合：

- 保存 Markdown 和代码；
- 记录历史版本；
- 恢复误删或错误修改；
- 多人协作；
- 自动生成和发布网站。

GitHub 不是普通的实时同步网盘。电脑之间同步通常需要：

```powershell
git push
```

以及：

```powershell
git pull
```

### OneDrive

OneDrive 是微软的云盘，可以理解为 Windows 版 iCloud。

文件放进 OneDrive 文件夹后，会自动上传和同步，不需要 Git 命令。

两者可以同时使用，但对于当前的 Markdown 知识库：

```
本地文件 + Git + GitHub 私有仓库
```

已经能够完成备份和版本管理。

---

## 三、是否可以全部放在 GitHub

可以。

建议至少把仓库设置为：

```
Private
```

这样只有自己可以访问。

但如果要使用 GitHub Pages 公开分享知识网站，可以采用两个仓库：

```
polar-notes-private
```

保存：

- 全部原始笔记；
- 草稿；
- 课程资料；
- 个人计划；
- 不适合公开的内容。

另一个：

```
polar-notes
```

保存：

- 整理完成的知识；
- 准备公开分享的文章；
- 个人作品集内容。

也就是：

```
私人总知识库
      ↓
挑选和整理
      ↓
公开知识网站
```

这是比给 GitHub Pages 网站加密码更简单、可靠的方式。

---

## 四、每个工具到底负责什么

### 1. Markdown

Markdown 是真正保存知识的文件格式，例如：

```
MLP.md
Git.md
NumPy广播机制.md
```

它不依赖某一个软件，未来即使不用 Obsidian 或 MkDocs，文件仍然可以阅读。

---

### 2. Obsidian

Obsidian主要负责：

- 写笔记；
- 浏览 Markdown；
- 双向链接；
- 搜索知识；
- 管理个人知识库。

它不是网站生成器。

---

### 3. VS Code

VS Code 在这个项目中主要负责：

- 编辑 Markdown；
- 修改 `mkdocs.yml`；
- 创建 `.gitignore`、`requirements.txt`；
- 创建 GitHub Actions 配置；
- 使用终端执行 Git 和 MkDocs 命令。

它也不是网站生成器。

可以这样分工：

```
Obsidian：主要写笔记
VS Code：管理网站项目和配置
```

实际上只用 VS Code 也完全可以。

---

### 4. Git

Git 负责记录文件变化。

例如它会记录：

```
新增了哪几行
删除了哪几行
哪个文件被修改
每一次提交时项目是什么状态
```

所以 Git 更像一个版本管理系统或时光机。

---

### 5. GitHub

GitHub 负责保存远程 Git 仓库。

它不只是网盘，还提供：

- GitHub Actions；
- GitHub Pages；
- Issues；
- Pull Requests；
- Wiki 等功能。

仓库主要保存的是：

```
Markdown 源文件
配置文件
Git 历史
自动化说明文件
```

---

### 6. MkDocs

MkDocs 是真正把 Markdown 转换成网页的软件。

例如：

```
MLP.md
   ↓
MkDocs
   ↓
MLP/index.html
```

浏览器最终阅读的是 HTML，不是原始 Markdown。

---

### 7. Material for MkDocs

Material 是 MkDocs 的主题。

它负责网站的外观和部分界面功能，例如：

- 顶部栏；
- 左侧导航；
- 搜索；
- 深色模式；
- 手机适配；
- 代码高亮；
- 页面目录。

关系是：

```
MkDocs：生成网站
Material：决定网站长什么样
```

Material 不是 GitHub 插件。

---

### 8. GitHub Actions

GitHub Actions 是 GitHub 提供的自动化计算服务。

每次你执行：

```powershell
git push
```

GitHub 可以根据配置文件：

```
.github/workflows/deploy.yml
```

临时创建一台 Ubuntu 虚拟机，然后执行：

```
下载仓库
安装 Python
安装 requirements.txt 中的软件
运行 MkDocs
生成 HTML
发布 HTML
销毁虚拟机
```

GitHub 并不是自己懂得如何生成 MkDocs 网站，而是临时提供一台电脑，让 MkDocs 在上面运行。

---

### 9. GitHub Pages

GitHub Pages 负责托管已经生成好的 HTML。

它本身不负责把 Markdown 转成网页。

仓库地址和网站地址不同：

```
仓库：
https://github.com/用户名/polar-notes
```

```
网站：
https://用户名.github.io/polar-notes/
```

别人访问网站地址时，看到的是最终网页，而不是 GitHub 文件列表。

---

## 五、`serve`、`build` 和 `gh-deploy` 的区别

### 本地预览

```powershell
mkdocs serve
```

它会：

- 启动本地服务器；
- 监听 `docs` 和 `mkdocs.yml`；
- 文件修改后自动重建；
- 浏览器自动刷新。

网站地址通常是：

```
http://127.0.0.1:8000/
```

这个地址只能在你自己的电脑上访问。

### 生成静态网页

```powershell
mkdocs build
```

它会一次性生成：

```
site/
```

其中包含 HTML、CSS、JavaScript 等静态文件。

生成结束后程序退出。

### 生成并发布

```powershell
mkdocs gh-deploy --force
```

它会：

1. 构建网站；
2. 把生成结果推送到 `gh-pages` 分支。

GitHub Actions 自动部署时通常使用这一条。

---

## 七、本地搭建完整步骤

### 第一步：创建 Conda 环境

打开 Anaconda Prompt、Miniconda Prompt 或已经支持 Conda 的 PowerShell。

```powershell
conda create -n polar-notes python=3.12 -y
```

激活：

```powershell
conda activate polar-notes
```

查看当前环境：

```powershell
conda env list
```

终端前面出现：

```
(polar-notes)
```

表示激活成功。

---

### 第二步：安装 MkDocs Material

```powershell
python -m pip install mkdocs-material
```

检查：

```powershell
mkdocs --version
```

安装 Material 时，MkDocs 会作为依赖一起安装。

---

### 第三步：创建项目文件夹

你的实际位置是：

```
C:\PolarNotes
```

创建并进入：

```powershell
mkdir C:\PolarNotes
cd C:\PolarNotes
```

---

### 第四步：初始化 MkDocs 项目

```powershell
mkdocs new .
```

最后的 `.` 表示：

> 在当前目录初始化项目。

生成：

```
C:\PolarNotes
├── docs
│   └── index.md
└── mkdocs.yml
```

---

### 第五步：用 VS Code 打开

```powershell
code .
```

VS Code 打开的是 `C:\PolarNotes` 整个项目文件夹。

---

### 第六步：配置 Material 主题

打开 `mkdocs.yml`：

```yaml
site_name: Polar Notes
site_description: Polar 的个人学习笔记

theme:
  name: material
  language: zh
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - search.suggest
    - search.highlight
```

YAML 使用空格缩进，不能随意使用 Tab。

---

### 第七步：编辑首页

打开：

```
docs/index.md
```

例如写：

```markdown
# Polar Notes

欢迎来到我的个人学习笔记。

## 主要内容

- 人工智能
- 数学
- 机械工程
- 各种零碎知识和教学

这是我的知识库网站。
```

---

### 第八步：启动本地网站

```powershell
mkdocs serve
```

打开：

```
http://127.0.0.1:8000/
```

终端出现：

```
Watching paths for changes
Serving on http://127.0.0.1:8000/
```

说明网站已经正常运行。

Material 曾显示一段关于 MkDocs 2.0 的警告，但当时网站仍然成功构建。这类信息不是当前项目报错，只要最后出现 `Serving on` 就说明本地服务器已经启动。

---

## 八、如何增加新文章

例如创建：

```
docs/
├── index.md
└── AI/
    ├── index.md
    └── MLP.md
```

`docs/AI/MLP.md`：

````markdown
# 多层感知机 MLP

多层感知机由多个线性层和非线性激活函数组成。

## 基本结构

```text
输入
→ Linear
→ ReLU
→ Linear
→ 输出
```

## 线性层

$$
Y = XW + b
$$
````

`AI/index.md` 可以作为 AI 分类的首页：

```markdown
# 人工智能

这里保存我的人工智能学习笔记。
```

---

## 九、导航有两种管理方式

### 方法一：在 `mkdocs.yml` 中手写

```yaml
nav:
  - 首页: index.md
  - 人工智能:
      - AI 概览: AI/index.md
      - 多层感知机: AI/MLP.md
```

网站导航顺序完全由这里决定，而不是由 Windows 文件夹顺序决定。

优点：

- 顺序明确；
- 控制精确；
- 适合文章数量较少的网站。

缺点：

- 文章多后需要反复修改。

---

### 方法二：自动导航插件

安装：

```powershell
python -m pip install mkdocs-awesome-nav
```

`mkdocs.yml`：

```yaml
plugins:
  - search
  - awesome-nav
```

注意：手动写出 `plugins` 后，需要显式保留：

```yaml
- search
```

否则默认搜索插件可能不会加载。

插件会根据：

```
docs/
```

自动生成导航。

还可以在目录中使用：

```
.nav.yml
```

控制名称和顺序。

例如：

```
docs/.nav.yml
```

```yaml
nav:
  - 首页: index.md
  - 人工智能: AI
  - 数学: Math
```

自动导航不是必需的，前期也可以先手写 `nav`。

---

## 十、`requirements.txt` 是什么

`requirements.txt` 不是 MkDocs 自动创建的，需要自己在项目根目录新建。

路径：

```
C:\PolarNotes\requirements.txt
```

内容：

```
mkdocs-material
mkdocs-awesome-nav
```

它只是依赖清单，不是软件本体。

GitHub Actions 会运行：

```powershell
python -m pip install -r requirements.txt
```

从而知道需要安装什么。

如果没有使用自动导航插件，就只写：

```
mkdocs-material
```

原则是：

> `mkdocs.yml` 使用了什么插件，`requirements.txt` 就必须包含对应的软件包。

本地虽然使用 Conda，GitHub Actions 仍然可以直接用 pip。两者不冲突。

---

## 十一、`environment.yml` 是什么

这是给你自己恢复 Conda 环境使用的，可选。

例如：

```yaml
name: polar-notes

dependencies:
  - python=3.12
  - pip
  - pip:
      - mkdocs-material
      - mkdocs-awesome-nav
```

以后换电脑，可以执行：

```powershell
conda env create -f environment.yml
```

重新创建环境。

两份文件的用途不同：

```
environment.yml
→ 给你自己的 Conda 使用
```

```
requirements.txt
→ 主要给 GitHub Actions 和普通 pip 环境使用
```

---

## 十二、`.gitignore` 是什么

`.gitignore` 告诉 Git：

> 哪些文件不要纳入版本管理，也不要上传 GitHub。

在 VS Code 中：

1. 右键项目根目录 `PolarNotes`；
2. 选择 New File；
3. 输入：

```
.gitignore
```

注意开头有一个点，没有 `.txt` 后缀。

内容：

```gitignore
site/
__pycache__/
*.pyc
.DS_Store
```

含义：

```
site/
```

忽略 MkDocs 自动生成的网站文件。

```
__pycache__/
*.pyc
```

忽略 Python 缓存。

```
.DS_Store
```

忽略 macOS 自动生成的文件。

核心思想是：

```
docs/、mkdocs.yml
是源文件，需要保存
```

```
site/
是自动生成文件，可以重新生成
```

---

## 十三、推荐的项目结构

```
PolarNotes/
├── .github/
│   └── workflows/
│       └── deploy.yml
├── docs/
│   ├── index.md
│   ├── AI/
│   │   ├── index.md
│   │   ├── LLM/
│   │   ├── DeepLearning/
│   │   └── PyTorch/
│   ├── CS/
│   │   ├── Python/
│   │   ├── Git/
│   │   ├── Linux/
│   │   └── DataStructure/
│   ├── Math/
│   │   ├── Calculus/
│   │   ├── LinearAlgebra/
│   │   ├── Probability/
│   │   └── Optimization/
│   ├── Mechanical/
│   │   ├── CAD/
│   │   ├── CAE/
│   │   └── FEM/
│   ├── Projects/
│   ├── Language/
│   └── Assets/
├── .gitignore
├── environment.yml
├── mkdocs.yml
└── requirements.txt
```

不建议一开始分太多层。可以先建立主要大类，等文章真的增多后再细分。

---

## 十四、图片如何保存

建议统一放在：

```
docs/Assets/
```

例如：

```
docs/Assets/relu.png
```

在 Markdown 中引用：

```markdown
![](../Assets/relu.png)
```

但如果笔记层级很深，相对路径会变得麻烦，也可以按主题建立图片目录，例如：

```
docs/Assets/AI/
docs/Assets/Math/
docs/Assets/Mechanical/
```

不要随意把图片散落在项目各处。

---

## 十五、初始化 Git

先停止本地服务器：

```
Ctrl + C
```

确认位置：

```powershell
cd C:\PolarNotes
```

初始化：

```powershell
git init
```

查看文件：

```powershell
git status
```

加入暂存区：

```powershell
git add .
```

第一次提交：

```powershell
git commit -m "Initialize Polar Notes"
```

如果要求设置身份：

```powershell
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub邮箱"
```

然后重新提交。

---

## 十六、创建 GitHub 仓库

在 GitHub 点击：

```
右上角 +
→ New repository
```

仓库名：

```
polar-notes
```

作为公开分享版本时选择：

```
Public
```

因为本地已经有项目，因此不要额外勾选：

```
Add a README
Add .gitignore
Choose a license
```

否则远程仓库会先产生一个与本地无关的提交，首次推送可能需要额外合并。

---

## 十七、本地连接 GitHub

先将主分支命名为 `main`：

```powershell
git branch -M main
```

添加远程仓库：

```powershell
git remote add origin https://github.com/Polar0216/polar-notes.git
```

检查：

```powershell
git remote -v
```

第一次推送：

```powershell
git push -u origin main
```

`-u` 会建立本地 `main` 和远程 `origin/main` 的跟踪关系。

以后只需：

```powershell
git push
```

---

## 十八、GitHub Actions 自动部署

创建：

```
C:\PolarNotes\.github\workflows\deploy.yml
```

内容：

```yaml
name: Deploy MkDocs

on:
  push:
    branches:
      - main

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: python -m pip install -r requirements.txt

      - name: Deploy website
        run: mkdocs gh-deploy --force
```

逐段含义：

```yaml
on:
  push:
    branches:
      - main
```

当 `main` 分支收到 Push 时运行。

```yaml
runs-on: ubuntu-latest
```

临时创建 Ubuntu 虚拟机。

```yaml
uses: actions/checkout@v4
```

把仓库内容下载到虚拟机。

```yaml
uses: actions/setup-python@v5
```

安装 Python。

```yaml
run: python -m pip install -r requirements.txt
```

安装 MkDocs、Material 和插件。

```yaml
run: mkdocs gh-deploy --force
```

生成网站并推送到 `gh-pages` 分支。

提交和推送：

```powershell
git add .
git commit -m "Add automatic deployment"
git push
```

---

## 十九、在哪里查看 Actions

在 GitHub 仓库顶部菜单可以看到：

```
Code
Issues
Pull requests
Actions
Projects
Wiki
Settings
```

`Actions` 就在顶部菜单中。

点击后可以看到工作流：

```
Deploy MkDocs
```

状态：

```
黄色圆点：正在运行
绿色对勾：成功
红色叉号：失败
```

你之前截图中顶部已经出现了 `Actions`，而首页提交旁边的黄色状态点说明工作流当时可能正在运行或等待结果。

如果完全没有工作流，检查：

```
.github/workflows/deploy.yml
```

路径是否正确，文件后缀是否真的是 `.yml`，以及 YAML 缩进是否正确。

---

## 二十、开启 GitHub Pages

进入：

```
Settings
→ Pages
```

选择：

```
Source: Deploy from a branch
```

然后：

```
Branch: gh-pages
Folder: / (root)
```

保存。

网站地址为：

```
https://Polar0216.github.io/polar-notes/
```

注意不是：

```
https://github.com/Polar0216/polar-notes
```

前者是网站，后者是仓库。

---

## 二十一、以后的日常工作流程

### 打开项目

```powershell
conda activate polar-notes
cd C:\PolarNotes
code .
```

需要本地预览时：

```powershell
mkdocs serve
```

打开：

```
http://127.0.0.1:8000/
```

### 更新网站

修改 `docs` 中的 Markdown 后：

```powershell
git add .
git commit -m "更新 NumPy 广播机制笔记"
git push
```

后续自动发生：

```
Push 到 GitHub
      ↓
GitHub Actions 创建 Ubuntu
      ↓
安装 requirements.txt
      ↓
运行 MkDocs
      ↓
更新 gh-pages
      ↓
GitHub Pages 更新网站
```

---

## 二十二、最核心的理解

整套系统可以浓缩为：

```
Markdown 是知识源文件
Git 是版本记录
GitHub 是远程仓库
GitHub Actions 是自动执行程序的云电脑
MkDocs 是 Markdown 到 HTML 的转换器
Material 是网站主题
GitHub Pages 是静态网页服务器
```

你真正需要长期维护的主要内容只有：

```
docs/
mkdocs.yml
requirements.txt
.github/workflows/deploy.yml
```

其中日常真正频繁修改的，基本只有：

```
docs/
```
