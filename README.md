# 医疗疾病诊断与数据采集系统 (Medical Disease Diagnosis & Data Collection System)

这是一个综合性的医疗数据科学项目，集成了多源医疗数据爬虫、基于BERT的疾病症状诊断模型以及可视化的Web交互界面。该系统旨在通过自动化手段收集高质量医疗数据，并利用深度学习技术提供智能化的疾病分诊建议。

## 🌟 核心功能 (Key Features)

1.  **多源数据采集 (Data Crawler)**
    *   支持从权威医疗网站（MedlinePlus, NHS.uk, ICD10Data）自动抓取疾病数据。
    *   包含数据清洗、去重、ICD编码补全等预处理功能。
    *   支持多种存储方式：JSON文件、MongoDB、MySQL。

2.  **智能疾病诊断 (Intelligent Diagnosis)**
    *   基于 **BERT (bert-base-chinese)** 预训练模型进行微调。
    *   能够根据用户输入的自然语言症状描述（如“头痛伴有恶心”），自动识别疾病类型。
    *   提供疾病类型预测、置信度分析及推荐就诊科室。

3.  **用户管理系统 (User Management)**
    *   提供完整的用户注册与登录功能。
    *   基于 SQLite 数据库存储用户信息，保障数据安全。
    *   支持用户会话管理。

4.  **Web 交互界面 (Web Interface)**
    *   基于 **Flask** 框架开发的轻量级Web应用。
    *   提供简洁的用户界面，支持症状输入与一键诊断。
    *   **历史记录功能**：本地存储用户的问诊历史，方便快速回溯。
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
│   ├── diagnose.html       # 诊断页面 (含历史记录)
│   ├── login.html          # 登录页面
│   └── register.html       # 注册页面
├── medical_train.py        # 模型训练脚本
├── medical_ui.py           # Web 应用启动脚本 (Flask入口)
├── medical_config.py       # 医疗模型相关配置
├── medical_data_loader.py  # 数据加载与预处理
├── models.py               # 数据库模型与用户管理逻辑
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
pip install torch transformers flask numpy requests beautifulsoup4 pymongo sqlalchemy pymysql werkzeug
```

### 3. 模型准备
项目默认使用 `bert-base-chinese`。如果本地没有模型文件，代码会自动尝试从 Hugging Face 下载，或者请手动将模型文件放入 `bert-base-chinese/` 目录。

### 4. 数据库初始化
首次运行前，系统会自动初始化 SQLite 数据库 (`medical_app.db`) 用于存储用户信息。

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

### 3. 进行智能诊断
*   在“智能诊断”页面，输入您的症状描述（例如：“最近胸痛，伴有呼吸困难”）。
*   点击“开始智能诊断”，系统将分析并返回：
    *   最可能的疾病类型及其置信度。
    *   推荐的就诊科室。
    *   详细的概率分布图表。

### 4. 训练诊断模型 (开发者选项)
如果需要使用自己的数据重新训练模型：
1.  准备数据：确保 `data/train_data.txt` 和 `data/val_data.txt` 存在且格式正确。
2.  运行训练脚本：
    ```bash
    python medical_train.py
    ```
    训练完成后，最佳模型将保存为 `best_disease_model/` 或覆盖指定路径。

### 5. 运行数据爬虫 (开发者选项)
如果需要抓取最新的医疗数据：
```bash
cd huank/DB_project/crawler
python main.py
```

## ⚙️ 配置说明 (Configuration)

*   **模型配置**: 修改 `medical_config.py` 可调整超参数（如 `EPOCHS`, `BATCH_SIZE`, `MAX_LEN`）及疾病标签映射。
*   **爬虫配置**: 修改 `huank/DB_project/crawler/config.py` 可调整爬取策略及存储路径。
*   **数据库配置**: 修改 `models.py` 可调整 SQLite 数据库文件路径。

## 💻 技术栈 (Tech Stack)

*   **编程语言**: Python
*   **深度学习**: PyTorch, Hugging Face Transformers (BERT)
*   **Web 框架**: Flask, Jinja2, Bootstrap 5
*   **数据采集**: Requests, BeautifulSoup, lxml
*   **数据库**: MongoDB, MySQL (爬虫数据), SQLite (用户数据)

---
*Generated for DS_Project1*
