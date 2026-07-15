# Git 与 GitHub 使用指南

下面按 **Windows + PowerShell + VS Code** 的环境讲。你可以一边看，一边创建一个测试项目实际操作。

---

## 一、Git 和 GitHub 到底是什么

### 1. Git 是本地版本管理工具

Git 安装在你的电脑上，用来记录项目每次修改。

例如你写代码时：

```
第一版：只能读取文件
第二版：增加 JSON 解析
第三版：增加异常处理
```

每完成一个阶段，就可以让 Git 保存一次记录。以后代码写坏了，可以查看之前的版本。

Git 更准确地说，不是简单保存“文件差异”，而是把每次提交看成项目在某个时刻的一张“快照”。

### 2. GitHub 是远程代码托管网站

GitHub 可以保存 Git 仓库，并提供：

- 云端备份；
- 多人协作；
- 查看代码；
- Issue 问题管理；
- Pull Request 代码审查；
- 开源项目发布。

因此：

```
Git：在你的电脑上管理版本
GitHub：把 Git 仓库放到网上
```

没有 GitHub，也可以使用 Git；但 GitHub 上的项目通常通过 Git 下载、更新和提交。

---

# 二、先建立 Git 的整体模型

初学 Git 最重要的是理解下面四个区域：

```
工作区
  ↓ git add
暂存区
  ↓ git commit
本地仓库
  ↓ git push
GitHub 远程仓库
```

## 1. 工作区

就是你平时看到的项目文件夹：

```
my-project/
├── main.py
├── README.md
└── data.json
```

你在 VS Code 里修改的文件，首先只存在于工作区。

## 2. 暂存区

暂存区表示：

> 下一次提交准备包含哪些修改。

执行：

```powershell
git add main.py
```

只是把 `main.py` 的当前修改放入暂存区。

**它没有上传到 GitHub，也没有正式生成版本。**

## 3. 本地仓库

执行：

```powershell
git commit -m "增加文件读取功能"
```

Git 才会在你电脑上创建一个正式版本记录。

## 4. 远程仓库

执行：

```powershell
git push
```

才会把本地提交上传到 GitHub。

所以一定要分清：

```
git add     选择要提交的内容
git commit  在本地保存版本
git push    上传到 GitHub
```

---

# 三、安装 Git

在 Git 官方网站安装 **Git for Windows**。安装后可以在以下终端使用 Git：

- PowerShell；
- Windows Terminal；
- VS Code 内置终端；
- Git Bash。

GitHub 官方建议先安装 Git、设置提交者姓名和邮箱，再选择 HTTPS 或 SSH 连接 GitHub。

安装完成后打开 PowerShell：

```powershell
git --version
```

出现类似：

```
git version 2.x.x
```

就说明安装成功。

这里是直接在 PowerShell 中输入命令，不是在 Python 解释器的：

```
>>>
```

提示符中输入。

---

# 四、第一次配置 Git

Git 每次提交都要记录：

```
谁提交的
提交者邮箱是什么
```

配置姓名：

```powershell
git config --global user.name "你的名字"
```

例如：

```powershell
git config --global user.name "Polar"
```

配置邮箱：

```powershell
git config --global user.email "你的GitHub邮箱"
```

例如：

```powershell
git config --global user.email "example@qq.com"
```

查看配置：

```powershell
git config --global --list
```

或者分别查看：

```powershell
git config --global user.name
git config --global user.email
```

`--global` 表示这套配置作用于当前电脑上的所有 Git 仓库。GitHub 会根据提交邮箱关联你的账号和贡献记录，因此建议使用已经绑定到 GitHub 的邮箱，或者 GitHub 提供的 `noreply` 邮箱。

建议再设置默认分支名称：

```powershell
git config --global init.defaultBranch main
```

将 VS Code 设置为 Git 默认编辑器：

```powershell
git config --global core.editor "code --wait"
```

GitHub 官方文档也提供了这项 VS Code 编辑器配置。

---

# 五、第一次使用 Git：创建本地仓库

## 1. 创建练习文件夹

```powershell
mkdir git-demo
cd git-demo
```

此时 PowerShell 当前所在位置就是：

```
git-demo
```

Git 命令通常需要在项目文件夹中执行。

## 2. 初始化 Git 仓库

```powershell
git init
```

执行后，Git 会在项目中创建一个隐藏目录：

```
.git
```

这个 `.git` 目录保存：

- 提交历史；
- 分支信息；
- 暂存区；
- 远程仓库地址；
- Git 配置。

因此，一个普通文件夹是否为 Git 仓库，关键就在于它是否包含 `.git`。

不要随便删除 `.git`，否则项目文件还在，但 Git 历史会消失。

## 3. 创建一个文件

PowerShell 中可以执行：

```powershell
"# Git Demo" | Out-File -Encoding utf8 README.md
```

或者直接用 VS Code 创建：

```
README.md
```

内容：

```markdown
# Git Demo

这是我的第一个 Git 项目。
```

## 4. 查看状态

```powershell
git status
```

此时可能显示：

```
Untracked files:
  README.md
```

`Untracked` 表示：

> 这个文件存在，但 Git 还没有开始追踪它。

## 5. 加入暂存区

```powershell
git add README.md
```

再次查看：

```powershell
git status
```

会看到：

```
Changes to be committed:
  new file: README.md
```

## 6. 创建第一次提交

```powershell
git commit -m "创建项目说明文件"
```

其中：

```
commit    创建一次版本记录
-m        直接填写提交说明
```

到这里，版本只是保存在你的电脑里，还没有上传 GitHub。

---

# 六、最常用的本地 Git 工作流程

以后每次修改代码，基本都在重复下面的过程。

## 第一步：修改文件

例如修改：

```
main.py
```

## 第二步：查看状态

```powershell
git status
```

这个命令非常重要。遇到不知道下一步做什么时，先执行：

```powershell
git status
```

## 第三步：查看具体改动

```powershell
git diff
```

它会显示：

```
哪些行被删除
哪些行被增加
```

`git diff` 用于比较工作区和暂存区之间的变化。

## 第四步：加入暂存区

只添加一个文件：

```powershell
git add main.py
```

添加多个文件：

```powershell
git add main.py README.md
```

添加当前目录全部修改：

```powershell
git add .
```

初学时不要无脑使用 `git add .`，因为它可能把数据集、临时文件甚至密码文件一起加入。

## 第五步：检查暂存内容

```powershell
git diff --staged
```

它显示下一次提交具体会包含什么。

## 第六步：提交

```powershell
git commit -m "实现用户登录功能"
```

提交信息应说明“做了什么”，不要总写：

```
修改
更新
123
test
```

更好的形式是：

```
增加 JSON 配置读取功能
修复空文件导致的异常
重构 Linear 层反向传播
补充项目安装说明
```

## 第七步：查看提交记录

```powershell
git log
```

简洁显示：

```powershell
git log --oneline
```

例如：

```
a91df23 增加异常处理
45c209a 实现文件读取
a0149ef 初始化项目
```

`git log` 是查看提交历史的基本命令。

---

# 七、把本地项目上传到 GitHub

假设你的电脑上已经有一个项目：

```
my-ai-project/
├── main.py
├── model.py
└── README.md
```

## 1. 在 GitHub 创建仓库

进入 GitHub：

```
右上角“+”
→ New repository
```

填写：

```
Repository name：my-ai-project
```

可见性：

```
Public：所有人可见
Private：只有你和获得权限的人可见
```

如果本地项目已经完成了 `git init` 和第一次提交，创建 GitHub 仓库时，建议暂时不要勾选：

```
Add a README file
Add .gitignore
Choose a license
```

否则本地和远程可能各自拥有不同的第一次提交，引发历史冲突。GitHub 官方也建议导入已有本地仓库时不要预先初始化这些文件。

## 2. 复制 HTTPS 地址

类似：

```
https://github.com/你的用户名/my-ai-project.git
```

## 3. 进入项目目录

```powershell
cd D:\Code\my-ai-project
```

确认位置：

```powershell
pwd
```

## 4. 如果还没有初始化

```powershell
git init
git add .
git commit -m "初始化项目"
```

## 5. 连接 GitHub 仓库

```powershell
git remote add origin https://github.com/用户名/my-ai-project.git
```

这里：

```
remote：远程仓库
origin：这个远程仓库在本地的简称
```

`origin` 不是 GitHub 的特殊关键字，只是约定俗成的默认远程仓库名称。`git remote add origin <URL>` 会把远程地址与名称 `origin` 关联。

查看远程地址：

```powershell
git remote -v
```

## 6. 将分支命名为 main

```powershell
git branch -M main
```

## 7. 第一次上传

```powershell
git push -u origin main
```

含义：

```
push     上传提交
origin   上传到名为 origin 的远程仓库
main     上传 main 分支
-u       建立本地 main 和远程 main 的跟踪关系
```

设置完成后，以后通常只需要：

```powershell
git push
```

第一次执行时，Windows 可能弹出浏览器，让你登录并授权 GitHub。

GitHub 目前不支持直接使用账号密码进行 Git 命令认证；HTTPS 通常通过 Git Credential Manager、浏览器授权或令牌完成。Git for Windows 会提供 Credential Manager 支持。

---

# 八、从 GitHub 下载已有项目

这叫做 **克隆仓库**。

## 1. 找到仓库地址

在 GitHub 项目页面：

```
Code → HTTPS → 复制地址
```

例如：

```
https://github.com/user/project.git
```

## 2. 进入准备存放项目的父目录

例如：

```powershell
cd D:\Code
```

注意不要先创建同名项目目录。

## 3. 克隆

```powershell
git clone https://github.com/user/project.git
```

Git 会自动创建：

```
D:\Code\project
```

进入项目：

```powershell
cd project
```

查看状态：

```powershell
git status
```

`git clone` 不只是下载文件，它还会一并下载 Git 历史，并自动配置远程仓库地址。

---

# 九、日常同步 GitHub 的标准流程

假设项目已经连接到 GitHub。

每次开始工作前：

```powershell
git pull
```

修改代码后：

```powershell
git status
git diff
git add main.py
git commit -m "增加模型训练日志"
git push
```

完整流程：

```
git pull
    ↓
修改文件
    ↓
git status
    ↓
git diff
    ↓
git add
    ↓
git commit
    ↓
git push
```

## `pull` 是什么

```powershell
git pull
```

相当于：

```
先 git fetch
再把远程修改整合到当前分支
```

Git 官方文档明确说明，`git pull` 首先执行获取远程内容的操作，然后把相应远程分支整合到当前分支。

## `fetch` 和 `pull` 的区别

```powershell
git fetch
```

只把远程仓库的新信息下载下来，不直接修改你当前的工作文件。

```powershell
git pull
```

下载之后还会尝试整合到当前分支。

初学阶段可以使用：

```powershell
git pull
```

以后多人协作时，再进一步学习 `fetch`、`merge` 和 `rebase`。

---

# 十、`.gitignore`：哪些文件不要上传

项目中有些文件不应该提交，例如：

```
Python 虚拟环境
缓存文件
密码和密钥
临时输出
大型数据集
模型权重
```

在项目根目录创建：

```
.gitignore
```

Python 项目可以先写：

```gitignore
# Python 缓存
__pycache__/
*.pyc
*.pyo

# 虚拟环境
.venv/
venv/
env/

# 环境变量和密钥
.env
.env.*

# 测试和工具缓存
.pytest_cache/
.mypy_cache/

# 输出目录
logs/
outputs/
checkpoints/

# 系统文件
.DS_Store
Thumbs.db
```

然后提交：

```powershell
git add .gitignore
git commit -m "添加 Git 忽略规则"
```

### 已经提交的文件不会自动被忽略

例如 `.venv` 已经被 Git 追踪，后来才写入 `.gitignore`，还需要执行：

```powershell
git rm -r --cached .venv
git commit -m "停止追踪虚拟环境"
```

这里 `--cached` 表示：

> 从 Git 的追踪记录中移除，但保留电脑上的实际文件。

---

# 十一、分支是什么

分支可以理解为一条独立的开发路线。

例如：

```
main
  ├── 登录功能分支
  ├── 数据读取分支
  └── 模型优化分支
```

主分支：

```
main
```

通常保存相对稳定的代码。

你需要开发新功能时，创建一个新分支：

```powershell
git switch -c feature-json-loader
```

含义：

```
switch       切换分支
-c           创建新分支
feature-...  分支名称
```

查看分支：

```powershell
git branch
```

带有星号的是当前分支：

```
* feature-json-loader
  main
```

在新分支中修改并提交：

```powershell
git add .
git commit -m "实现 JSON 数据加载器"
```

第一次上传该分支：

```powershell
git push -u origin feature-json-loader
```

Git 官方术语中，分支代表一条开发路线；当前分支的最新提交位置会随着新提交向前移动。

---

# 十二、Pull Request 是什么

把分支上传 GitHub 后，可以创建 Pull Request，简称 PR。

它表达的是：

> 我在这个分支完成了一些修改，希望把它合并进 main。

流程：

```
创建分支
→ 修改代码
→ commit
→ push
→ 在 GitHub 创建 Pull Request
→ 审查代码
→ Merge
```

GitHub 把 Pull Request 作为协作流程的核心：它展示两个分支之间的差异，允许讨论、审查并最终合并修改。

合并完成后，本地切回 `main`：

```powershell
git switch main
git pull
```

删除本地旧分支：

```powershell
git branch -d feature-json-loader
```

---

# 十三、发生代码冲突怎么办

假设你和另一个人同时修改了同一文件的同一位置。

执行：

```powershell
git pull
```

可能出现：

```
CONFLICT (content): Merge conflict in main.py
```

打开文件会看到：

```
<<<<<<< HEAD
print("我的代码")
=======
print("远程代码")
>>>>>>> origin/main
```

含义：

```
<<<<<<< HEAD
你本地的内容

=======
分界线

>>>>>>> origin/main
远程分支的内容
```

你需要手动决定最终保留什么，例如：

```python
print("合并后的代码")
```

并删除这些标记：

```
<<<<<<<
=======
>>>>>>>
```

然后执行：

```powershell
git add main.py
git commit -m "解决 main.py 合并冲突"
git push
```

---

# 十四、撤销修改

Git 的撤销操作很多，初学阶段记住下面几种就够了。

## 1. 文件改坏了，但还没 `git add`

```powershell
git restore main.py
```

作用：

> 放弃 `main.py` 尚未暂存的修改，恢复到暂存区中的状态。

注意：修改通常会直接丢失。

## 2. 已经 `git add`，但不想提交

```powershell
git restore --staged main.py
```

作用：

> 把文件移出暂存区，但保留文件中的修改。

## 3. 最近一次提交信息写错

```powershell
git commit --amend -m "正确的提交说明"
```

如果只是忘记添加一个文件：

```powershell
git add forgotten.py
git commit --amend --no-edit
```

这会重写最后一次提交，因此已经推送并与他人共享的提交不要随意 `amend`。

## 4. 撤销已经推送的提交

先查看提交：

```powershell
git log --oneline
```

然后：

```powershell
git revert 提交编号
```

例如：

```powershell
git revert a91df23
```

`revert` 不会删除旧历史，而是创建一个新的提交，用来抵消指定提交的效果，因此更适合已经推送和共享的历史。

## 5. 谨慎使用

```powershell
git reset --hard
```

这个命令可能直接丢弃工作区和暂存区修改。除非明确知道自己在做什么，否则不要复制网上的 `reset --hard` 命令。

---

# 十五、参与别人的 GitHub 开源项目

通常使用下面的流程：

```
Fork
→ Clone
→ 创建分支
→ 修改
→ Commit
→ Push
→ Pull Request
```

## 1. Fork

在别人仓库页面点击：

```
Fork
```

它会在你的 GitHub 账号下创建一个副本。

## 2. 克隆自己的 Fork

```powershell
git clone https://github.com/你的用户名/project.git
cd project
```

## 3. 添加原项目为 upstream

```powershell
git remote add upstream https://github.com/原作者/project.git
```

查看：

```powershell
git remote -v
```

可能显示：

```
origin    你的仓库
upstream  原作者仓库
```

## 4. 创建功能分支

```powershell
git switch -c fix-file-reader
```

## 5. 修改并上传

```powershell
git add .
git commit -m "修复空文件读取错误"
git push -u origin fix-file-reader
```

然后在 GitHub 上向原项目提交 Pull Request。

---

# 十六、README 应该写什么

`README.md` 是项目说明文件，通常包括：

````markdown
# 项目名称

一句话介绍项目。

## 功能

- 功能一
- 功能二

## 安装

```bash
pip install -r requirements.txt
````

## 使用

```bash
python main.py
```

## 项目结构

```
project/
├── main.py
├── model.py
└── README.md
```

## License

MIT

````
GitHub 通常会自动展示仓库根目录中的 README。README 常用于说明项目作用、使用方法、如何开始以及维护者信息。:contentReference[oaicite:15]{index=15}

---

# 十七、常见报错

## 1. `not a git repository`

```text
fatal: not a git repository
````

说明当前目录不是 Git 仓库。

先查看当前位置：

```powershell
pwd
```

再进入正确项目：

```powershell
cd D:\Code\my-project
```

或者如果这是新项目：

```powershell
git init
```

---

## 2. `Author identity unknown`

说明没有配置姓名或邮箱：

```powershell
git config --global user.name "Polar"
git config --global user.email "你的邮箱"
```

---

## 3. `remote origin already exists`

说明已经存在名为 `origin` 的远程地址。

查看：

```powershell
git remote -v
```

修改地址：

```powershell
git remote set-url origin 新地址
```

不要重复执行：

```powershell
git remote add origin ...
```

---

## 4. `Authentication failed`

可能是 GitHub 登录信息过期。

Windows 中可以打开：

```
控制面板
→ 用户账户
→ 凭据管理器
→ Windows 凭据
```

找到旧的 GitHub 凭据并删除，再次 `git push` 重新登录。这也是 GitHub 官方给出的 Windows 凭据重置方式。

---

## 5. `rejected non-fast-forward`

通常表示：

> GitHub 上存在你本地没有的提交。

先尝试：

```powershell
git pull
```

处理完合并或冲突后：

```powershell
git push
```

不要看到报错就立刻使用：

```powershell
git push --force
```

强制推送可能覆盖远程历史。

---

# 十八、Git 命令速查

| 命⁠令 | 作⁠用 |
| --- | --- |
| `git init` | 把⁠当⁠前⁠文⁠件⁠夹⁠初⁠始⁠化⁠为 Git 仓⁠库 |
| `git clone URL` | 克⁠隆⁠远⁠程⁠仓⁠库 |
| `git status` | 查⁠看⁠仓⁠库⁠状⁠态 |
| `git diff` | 查⁠看⁠未⁠暂⁠存⁠修⁠改 |
| `git diff --staged` | 查⁠看⁠已⁠暂⁠存⁠修⁠改 |
| `git add 文⁠件` | 把⁠修⁠改⁠放⁠入⁠暂⁠存⁠区 |
| `git add .` | 暂⁠存⁠当⁠前⁠目⁠录⁠全⁠部⁠修⁠改 |
| `git commit -m "说⁠明"` | 创⁠建⁠本⁠地⁠提⁠交 |
| `git log --oneline` | 查⁠看⁠简⁠洁⁠提⁠交⁠记⁠录 |
| `git remote -v` | 查⁠看⁠远⁠程⁠仓⁠库 |
| `git pull` | 获⁠取⁠并⁠整⁠合⁠远⁠程⁠修⁠改 |
| `git push` | 上⁠传⁠本⁠地⁠提⁠交 |
| `git branch` | 查⁠看⁠分⁠支 |
| `git switch 分⁠支` | 切⁠换⁠分⁠支 |
| `git switch -c 分⁠支` | 创⁠建⁠并⁠切⁠换⁠分⁠支 |
| `git merge 分⁠支` | 合⁠并⁠分⁠支 |
| `git restore 文⁠件` | 放⁠弃⁠未⁠暂⁠存⁠修⁠改 |
| `git restore --staged 文⁠件` | 取⁠消⁠暂⁠存 |
| `git revert 提⁠交` | 创⁠建⁠反⁠向⁠提⁠交 |
| `git stash` | 临⁠时⁠收⁠起⁠未⁠提⁠交⁠修⁠改 |

Git 官方命令参考将这些命令分别归入仓库创建、快照、分支合并和远程共享等类别。

---

# 十九、你现在最应该练习的一套流程

创建一个简单的 Python 项目：

```
git-practice/
├── main.py
├── README.md
└── .gitignore
```

然后完整执行一次：

```powershell
cd D:\Code
mkdir git-practice
cd git-practice

git init

"print('Hello Git')" | Out-File -Encoding utf8 main.py
"# Git Practice" | Out-File -Encoding utf8 README.md

git status
git add main.py README.md
git commit -m "初始化 Python 项目"
```

在 GitHub 创建一个空仓库后：

```powershell
git remote add origin https://github.com/你的用户名/git-practice.git
git branch -M main
git push -u origin main
```

再修改 `main.py`：

```python
def main():
    print("Hello Git and GitHub")

if __name__ == "__main__":
    main()
```

然后：

```powershell
git status
git diff
git add main.py
git diff --staged
git commit -m "将程序入口封装为 main 函数"
git push
```

只要你能独立完成这一遍，就已经掌握了个人项目中最常用的 Git/GitHub 主流程：

```
修改 → add → commit → push
```