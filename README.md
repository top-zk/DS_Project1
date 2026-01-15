# 医疗疾病诊断与数据采集系统 (Medical Disease Diagnosis & Data Collection System)

这是一个综合性的医疗数据科学项目，集成了多源医疗数据爬虫、基于BERT的疾病症状诊断模型以及可视化的Web交互界面。该系统旨在通过自动化手段收集高质量医疗数据，并利用深度学习技术提供智能化的疾病分诊建议，同时提供丰富的医疗百科知识库。

## 🌟 核心功能 (Key Features)

1.  **智能问答咨询 (Intelligent Chat Consultation)** (New!)
    *   **多轮对话交互**：提供类似聊天机器人的交互体验，支持连续对话。
    *   **动态追问**：根据初步识别的症状类型（如感冒、心脏问题等），智能抛出针对性的追问（如“体温多少？”“胸痛持续多久？”），以获取更精确的诊断信息。
    *   **智能报告生成**：对话结束后，自动生成包含初步诊断、推荐科室、详细说明及免责声明的完整报告。
    *   **上下文记忆**：能够结合上下文理解用户的症状描述。

2.  **多源数据采集 (Data Crawler)**
    *   支持从权威医疗网站（MedlinePlus, NHS.uk, ICD10Data）自动抓取疾病数据。
    *   包含数据清洗、去重、ICD编码补全等预处理功能。
    *   支持多种存储方式：JSON文件、MongoDB、MySQL。

3.  **智能疾病诊断 (Intelligent Diagnosis)**
    *   **混合诊断引擎**：结合 **BERT (bert-base-chinese)** 深度学习模型与 **专家规则系统 (Rule-based Keyword Matching)**。
    *   **高精度识别**：利用关键词加权机制，显著提升了对典型症状（如“头痛”、“骨折”）的诊断准确率，有效避免误诊。
    *   能够根据用户输入的自然语言症状描述（如“头痛伴有恶心”），自动识别疾病类型。
    *   **智能校验**：前端与后端双重校验，确保输入的症状描述足够详细（至少10字），以提高诊断准确率。
    *   提供疾病类型预测、置信度分析及推荐就诊科室。

4.  **医疗百科 (Medical Encyclopedia)**
    *   **知识库构建**：基于爬取的数据自动构建本地医疗百科数据库 (SQLite)。
    *   **结构化数据展示 (New!)**：对核心疾病数据进行了深度结构化清洗，详情页提供“核心方案”与“详细说明”的分层展示，重点突出，易于阅读。
    *   **A-Z 索引**：支持按疾病名称首字母（A-Z）快速浏览查找。
    *   **全文搜索**：支持通过关键词搜索疾病名称或症状，配备**全新设计的搜索交互界面**。
    *   **详情展示**：提供疾病的详细介绍、症状描述等信息。

5.  **用户管理系统 (User Management)**
    *   提供完整的用户注册与登录功能。
    *   基于 SQLite 数据库存储用户信息，保障数据安全。
    *   支持用户会话管理。

6.  **Web 交互界面 (Web Interface)**
    *   基于 **Flask** 框架开发的轻量级Web应用。
    *   **UI/UX 全面升级**：采用现代化设计语言，统一视觉风格。新增卡片悬停、按钮微交互及平滑过渡动画，提供流畅的用户体验。
    *   **历史记录功能**：本地存储用户的问诊历史，支持侧边栏快速查看与回填。
    *   实时展示诊断结果、概率分布及就诊建议。

## 📂 项目结构 (Project Structure)

```
DS_Project1/
├── bert-base-chinese/      # BERT 预训练模型文件 (需下载或放置在此)
├── data/                   # 数据集目录
│   ├── train_data.txt      # 训练数据
│   └── val_data.txt        # 验证数据
├── huank/DB_project/crawler/ # 爬虫模块核心代码
│   ├── main.py             # 爬虫入口程序
│   ├── sites/              # 各站点爬虫实现 (medlineplus, nhsuk, icd10data)
│   ├── storage.py          # 数据存储逻辑 (JSON, Mongo, MySQL)
│   └── ...
├── templates/              # Web应用模板
│   ├── base.html           # 基础布局模板
│   ├── home.html           # 首页
│   ├── chat.html           # 智能问答咨询页面 (New!)
│   ├── diagnose.html       # 诊断页面 (含历史记录与输入校验)
│   ├── encyclopedia.html   # 医疗百科首页 (搜索 & 索引)
│   ├── encyclopedia_detail.html # 医疗百科详情页
│   ├── login.html          # 登录页面
│   └── register.html       # 注册页面
├── logs/                   # 系统日志目录
├── import_data.py          # 批量数据导入工具
├── medical_train.py        # 模型训练脚本
├── medical_ui.py           # Web 应用启动脚本 (Flask入口)
├── medical_config.py       # 医疗模型配置 (含关键词映射规则)
├── medical_data_loader.py  # 数据加载与预处理
├── models.py               # 数据库模型 (用户 & 百科)
├── requirements.txt        # 项目依赖清单
└── ...
```

## 🛠️ 安装与配置 (Installation & Setup)

### 1. 环境要求
*   Python 3.8+
*   CUDA (可选，用于GPU加速训练)

### 2. 安装依赖
请确保安装了项目所需的Python库：

```bash
pip install torch transformers flask numpy requests beautifulsoup4 pymongo sqlalchemy pymysql werkzeug pypinyin deep-translator
```

### 3. 模型准备
项目默认使用 `bert-base-chinese`。如果本地没有模型文件，代码会自动尝试从 Hugging Face 下载，或者请手动将模型文件放入 `bert-base-chinese/` 目录。

### 4. 数据库初始化
首次运行前，系统会自动初始化 SQLite 数据库 (`medical_app.db`)。

**数据导入 (Data Import)**:
如果您已经运行了爬虫并希望将数据导入到百科数据库中，请运行：
```bash
python import_data.py
```
该工具会自动扫描 `output/` 目录下的 JSON 数据并将其导入数据库，同时在 `logs/` 目录下生成导入报告。

## 🚀 使用指南 (Usage Guide)

### 1. 启动 Web 诊断系统
直接运行 `medical_ui.py` 启动 Flask 服务：

```bash
python medical_ui.py
```
启动后访问 `http://localhost:5000`。

### 2. 用户注册与登录
*   访问首页点击右上角“注册”按钮，创建新账号。
*   使用注册邮箱登录后，即可使用完整的诊断功能。

### 3. 智能问答咨询 (New!)
*   点击导航栏的“问答咨询”或首页相关入口。
*   **交互式诊断**：像聊天一样描述您的不适，系统会根据您的情况进行追问（例如：“是否发烧？”“疼痛持续多久？”）。
*   **获取报告**：回答完所有问题后，系统将为您生成一份详细的智能推测报告，包含初步诊断和就诊建议。

### 4. 进行传统智能诊断
*   在“智能诊断”页面，输入您的症状描述（例如：“最近经常感到头痛、头晕，晚上失眠...”）。
*   **注意**：为了保证诊断效果，请至少输入 **10个字** 的描述。
*   点击“开始智能诊断”，系统将分析并返回：
    *   最可能的疾病类型及其置信度。
    *   推荐的就诊科室。
    *   详细的概率分布图表。

### 5. 查阅医疗百科
*   点击顶部导航栏的“医疗百科”。
*   您可以直接在搜索框输入疾病名称，或点击下方的字母索引浏览疾病列表。
*   点击具体疾病可查看详细信息。

### 6. 训练诊断模型 (开发者选项)
如果需要使用自己的数据重新训练模型：
1.  准备数据：确保 `data/train_data.txt` 和 `data/val_data.txt` 存在且格式正确。
2.  运行训练脚本：
    ```bash
    python medical_train.py
    ```
    训练完成后，最佳模型将保存为 `best_disease_model/` 或覆盖指定路径。

### 7. 运行数据爬虫 (开发者选项)
如果需要抓取最新的医疗数据：
```bash
cd huank/DB_project/crawler
python main.py
```

## ⚙️ 配置说明 (Configuration)

*   **模型与规则配置**: 修改 `medical_config.py` 可调整超参数（如 `EPOCHS`, `BATCH_SIZE`）以及 **`DISEASE_SYMPTOM_KEYWORDS` (诊断关键词规则)**。
*   **爬虫配置**: 修改 `huank/DB_project/crawler/config.py` 可调整爬取策略及存储路径。
*   **数据库配置**: 修改 `models.py` 可调整 SQLite 数据库文件路径及百科数据导入逻辑。

## 💻 技术栈 (Tech Stack)

*   **编程语言**: Python
*   **深度学习**: PyTorch, Hugging Face Transformers (BERT)
*   **Web 框架**: Flask, Jinja2, Bootstrap 5
*   **数据采集**: Requests, BeautifulSoup, lxml
*   **数据库**: MongoDB, MySQL (爬虫数据), SQLite (用户数据 & 百科知识库)

---
*Generated for DS_Project1*
