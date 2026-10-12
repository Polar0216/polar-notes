# MkDocs 配置 LaTeX 数学公式总结

## 一、核心原理

MkDocs 默认不会直接渲染 LaTeX 数学公式，需要配置以下三个组件：

| 组⁠件                 | 作⁠用                                                 |
| --------------------- | ----------------------------------------------------- |
| `pymdownx.arithmatex` | 识⁠别⁠并⁠处⁠理 Markdown 中⁠的⁠数⁠学⁠公⁠式             |
| MathJax               | 在⁠浏⁠览⁠器⁠中⁠真⁠正⁠渲⁠染⁠数⁠学⁠公⁠式                |
| `mathjax.js`          | 配⁠置 MathJax，并⁠处⁠理⁠页⁠面⁠切⁠换⁠后⁠的⁠重⁠新⁠渲⁠染 |

只需配置一次，整个 MkDocs 网站的所有 Markdown 文件都能使用 LaTeX 数学公式。

## 二、文件目录结构

采用最简单的方案，将 `mathjax.js` 直接放在 `docs/` 根目录。

```纯文本
polar-notes/
├── mkdocs.yml
├── requirements.txt
└── docs/
    ├── mathjax.js          # 新增
    ├── index.md
    ├── 多元函数微分学/
    ├── 复变函数/
    ├── 电磁学/
    └── 理论力学/
```

注意：`mathjax.js` 不建议直接放在 `mkdocs.yml` 旁边，因为 MkDocs 默认只会将 `docs/` 中的静态资源发布到网站。

## 三、修改 mkdocs.yml

保留原有功能，在 `markdown_extensions` 中增加 Arithmatex，并添加 `extra_javascript`。

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
    - toc.follow

plugins:
  - search
  - awesome-nav

markdown_extensions:
  - toc:
      toc_depth: "2-6"
      permalink: true
  - pymdownx.arithmatex:
      generic: true

extra_javascript:
  - mathjax.js
  - https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js
```

## 四、创建 mathjax.js

在 `docs/mathjax.js` 中写入：

```javascript
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
```

此文件负责设置数学公式解析规则，并确保 Material for MkDocs 切换页面后能够重新渲染公式。

## 五、本地安装与测试

在项目使用的 Python 环境中执行：

```powershell
python -m pip install pymdown-extensions
python -m mkdocs serve
```

浏览器访问：

`http://127.0.0.1:8000`

确认公式正常显示后，运行：

```powershell
python -m mkdocs build
```

如果构建成功，就可以准备部署网站。

## 六、Markdown 公式写法

行内公式：

```latex
方向导数为 $D_{\mathbf v}f(a)$。
```

渲染效果：方向导数为 $D_{\mathbf v}f(a)$。

独立公式：

```latex
$$
D_{\mathbf v}f(a)
=
\nabla f(a)\cdot\mathbf v
$$
```

渲染效果：

$$
D_{\mathbf v}f(a)=\nabla f(a)\cdot\mathbf v
$$

## 七、同步到 GitHub 网站

本地确认无误后：

```powershell
git add mkdocs.yml docs/mathjax.js
git commit -m "Add LaTeX math support"
git push origin main
```

如果 GitHub Actions 已配置为在推送 `main` 分支后自动部署，那么等待部署完成，刷新网页即可。

如果没有自动部署，需要检查仓库的 GitHub Pages 设置及部署工作流。

## 八、几个注意事项

- 作用范围：`mkdocs.yml` 对整个 MkDocs 项目生效，不限于单个文件夹。
- 新增笔记：以后新增的 Markdown 文件同样可以直接使用数学公式。
- 编辑器区别：MkDocs 网站的渲染方式与 VS Code 的 Markdown 预览相互独立。
- 网络依赖：目前 MathJax 通过 `unpkg.com` 加载，网站访问者需要能连接该 CDN；以后可以改成本地托管。
- 部署依赖：如果 GitHub 构建环境缺少 `pymdown-extensions`，需要将其添加到 `requirements.txt`。

最终工作流程：

本地写 Markdown → MkDocs 识别公式 → 浏览器使用 MathJax 渲染 → Git 提交并推送 → GitHub Actions 构建发布 → 网站显示数学公式。

以后正常写笔记即可，不需要对每个文件重复配置。

## 补充

mkdocs新安装了东西，但是requirements没有更改呀，那github的action就没有改变，为什么网页会调整呢

你发现了一个非常关键的问题！这其实涉及 Python 依赖管理、GitHub Actions 和网站自动部署之间的关系。

你的理解是对的：本地通过 `pip install` 安装的软件包，并不会自动同步到 GitHub，`requirements.txt` 也不会自动更新。

但是有一种情况，即使不修改 `requirements.txt`，GitHub Actions 也能正常构建网站。

### 一、首先理解两个独立的 Python 环境

你的电脑

本地 Python 环境

pip install 安装依赖

GitHub Actions

独立的运行环境

根据工作流安装依赖

两边不会自动共享已安装的软件包

requirements.txt

用来记录项目需要安装哪些 Python 包

例如，你在本地执行：

```powershell
pip install pymdown-extensions
```

这只是在本地 Python 环境安装了扩展。

而 GitHub Actions 通常会执行类似：

```yaml
- name: Install dependencies
  run: pip install -r requirements.txt
```

它只会按照依赖文件及其间接依赖来安装。

所以，如果 GitHub 环境没有 Arithmatex 扩展，那么你修改了 `mkdocs.yml` 后，构建就可能失败。

### 二、为什么你的项目可能不需要修改 requirements.txt？

我刚刚查看了你的 [polar-notes 的 requirements.txt](https://github.com/Polar0216/polar-notes/blob/main/requirements.txt)，目前只有：

```纯文本
mkdocs-material
mkdocs-awesome-nav
```

关键在于：`mkdocs-material` 本身就依赖 `pymdown-extensions`。 安装前者时，pip 会自动安装后者。![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+1

依赖关系可以理解为：

```纯文本
requirements.txt
│
├── mkdocs-material
│   ├── mkdocs
│   ├── pymdown-extensions  ← 自动安装
│   └── 其他依赖
│
└── mkdocs-awesome-nav
```

所以，GitHub Actions 原本就可能已经安装了 Arithmatex，只是你此前没有在 `mkdocs.yml` 中启用它。

你之前执行的 `pip install pymdown-extensions` 很可能只是确认本地安装了这个扩展。

### 三、既然 Actions 没变，为什么网站会改变？

这是另一个很重要的概念：

GitHub Actions 的工作流可以不变，但它每次构建时读取的项目文件可以改变。

例如：

第一次构建

旧版本

安装依赖 → 读取旧 `mkdocs.yml` → 不启用 Arithmatex → 生成没有公式渲染功能的网站

第二次构建

新版本

安装同样的依赖 → 读取新 `mkdocs.yml` → 启用 Arithmatex 并引入 MathJax → 生成支持公式的网站

就像你写 Python 程序一样，Python 解释器不用改变，但是你修改了 `.py` 文件，重新运行后程序的行为就可以改变。

这里还要区分：

- `pymdown-extensions`：构建时运行的 Python 包，需要在 Actions 环境中安装。
- `mathjax.js`：随网站发布的静态资源，不需要 pip 安装。
- `mkdocs.yml`：决定构建时启用哪些功能。

当然，前提是你的 Actions 工作流会在代码推送后重新构建并部署网站。

### 四、那究竟要不要修改 requirements.txt？

对于你现在的项目，不修改也大概率可以正常工作。

不过我更推荐显式写出直接使用的依赖：

```纯文本
mkdocs-material
mkdocs-awesome-nav
pymdown-extensions
```

这样即使以后调整 Material 主题的依赖关系，也能更明确地保证数学扩展被安装。

你可以用下面的命令查看本地是否已经安装：

```powershell
python -m pip show pymdown-extensions
```

此外，不建议直接用 `pip freeze` 覆盖 `requirements.txt`，因为那会把本地环境中所有不相关的软件包也记录进去。

你这次的疑问实际上已经触及了 CI/CD（持续集成与持续部署） 的基本原理：

`requirements.txt` 决定需要什么环境，`mkdocs.yml` 决定如何构建，GitHub Actions 负责自动执行构建与部署。

它们彼此相关，但并不是同一个东西。
