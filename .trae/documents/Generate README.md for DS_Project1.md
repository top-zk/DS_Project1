Based on the analysis of your project structure and code, I have designed a comprehensive `README.md` file.

## Plan to Generate README.md

I will create a new `README.md` file in the root directory (`c:\Users\胡安康\Desktop\DS_Project1\`) with the following structure:

### 1. Project Title & Overview
- **Title**: Medical Disease Diagnosis & Data Collection System (医疗疾病诊断与数据采集系统)
- **Description**: A comprehensive system combining a medical data crawler, a BERT-based disease diagnosis model, and a Flask web interface.

### 2. Key Features (核心功能)
- **Data Crawler**: Automated collection of disease data from MedlinePlus, NHS.uk, and ICD10Data with data cleaning and multi-format storage (JSON, MongoDB, MySQL).
- **Intelligent Diagnosis**: BERT-based Deep Learning model (`bert-base-chinese`) to classify symptoms into medical departments.
- **Web Interface**: User-friendly Flask application for real-time symptom analysis and department recommendation.

### 3. Project Structure (项目结构)
- Explain key directories:
  - `huank/DB_project/crawler`: Data collection module.
  - `medical_train.py` & `medical_ui.py`: Core AI and Web logic.
  - `data/`: Training and validation datasets.
  - `bert-base-chinese/`: Pre-trained model files.

### 4. Installation & Setup (安装与配置)
- Python environment requirements (Python 3.8+).
- Dependency installation (`pip install -r requirements.txt`).
- Database setup (if applicable for the crawler).

### 5. Usage Guide (使用指南)
- **Data Crawling**: How to run `huank/DB_project/crawler/main.py`.
- **Model Training**: How to run `medical_train.py`.
- **Web Application**: How to launch `medical_ui.py` and access the interface.

### 6. Configuration (配置说明)
- Description of `medical_config.py` and crawler configurations.

### 7. Tech Stack (技术栈)
- **Core**: Python, PyTorch, Transformers (BERT).
- **Web**: Flask.
- **Data**: MongoDB, MySQL, BeautifulSoup, Pandas.

I will write this content in **Chinese** to match the project's primary language (as seen in the code comments and variable names).
