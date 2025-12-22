# config.py
import torch
import os

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型参数
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
MODEL_PATH = r"D:\DS_Project\bert-base-chinese"

# 数据文件路径
DATA_PATH = r"D:\question2.xlsx"
# 创建data文件夹路径
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
# 输出数据路径
TRAIN_DATA_PATH = "data/train_data.txt"
VAL_DATA_PATH = "data/val_data.txt"
ENCYCLOPEDIA_JSON_DIR = r"D:\DS_Project\web_crapper\web_scraper\encyclopedia_data"

# 标签映射
INTENT_TYPES = {
    0: '耳鼻喉科',
    1: '内科',
    2: '外科',
    3: '骨科',
    4: '妇科',
    5: '儿科',
    6: '9',
    7: '支付问题',
    8: '销售额问题',
    9: '库存问题'
}

# 样本数据（备用）
SAMPLE_DATA = [
    ("如何查询销售额", 8),
    ("怎么查看加油站销量", 8),
    ("柴油库存情况", 9),
    ("汽油库存统计", 9),
    ("历史销售数据查询", 8),
    ("帮我查一下上个月的总销售额", 8),
    ("新江加油站的库存情况", 9),
]