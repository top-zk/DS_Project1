from medical_ui import app
from models import db, SymptomDiseaseRule
import random

# Disease Data Pool
DISEASE_DATA = {
    "感冒": {
        "dept": "呼吸内科",
        "symptoms": ["咳嗽", "流鼻涕", "鼻塞", "喉咙痛", "发烧", "乏力", "打喷嚏", "头痛", "肌肉酸痛", "食欲不振"],
        "advice": "多喝水，注意休息，清淡饮食。如症状加重请及时就医。"
    },
    "流感": {
        "dept": "呼吸内科",
        "symptoms": ["高热", "寒战", "头痛", "肌肉酸痛", "乏力", "咳嗽", "咽痛", "流鼻涕", "胸痛", "呼吸急促"],
        "advice": "居家隔离，服用抗病毒药物，物理降温，密切监测体温。"
    },
    "急性胃肠炎": {
        "dept": "消化内科",
        "symptoms": ["腹痛", "腹泻", "恶心", "呕吐", "发烧", "腹胀", "食欲减退", "乏力", "脱水", "头晕"],
        "advice": "暂时禁食或流食，注意补液防止脱水，必要时使用抗生素。"
    },
    "高血压": {
        "dept": "心血管内科",
        "symptoms": ["头晕", "头痛", "心悸", "耳鸣", "视力模糊", "失眠", "肢体麻木", "颈项板滞", "乏力", "胸闷"],
        "advice": "低盐低脂饮食，规律服药，监测血压，保持心情舒畅。"
    },
    "冠心病": {
        "dept": "心血管内科",
        "symptoms": ["胸痛", "胸闷", "心悸", "气短", "乏力", "出汗", "恶心", "左肩放射痛", "呼吸困难", "头晕"],
        "advice": "随身携带急救药物，避免劳累和情绪激动，低脂饮食。"
    },
    "糖尿病": {
        "dept": "内分泌科",
        "symptoms": ["多饮", "多食", "多尿", "体重减轻", "视力模糊", "皮肤瘙痒", "乏力", "手脚麻木", "伤口难愈", "易感染"],
        "advice": "控制饮食，适量运动，规律监测血糖，遵医嘱用药。"
    },
    "偏头痛": {
        "dept": "神经内科",
        "symptoms": ["单侧头痛", "搏动性疼痛", "恶心", "呕吐", "畏光", "畏声", "头晕", "视力模糊", "疲劳", "易怒"],
        "advice": "避免诱发因素（如压力、特定食物），休息，服用止痛药。"
    },
    "颈椎病": {
        "dept": "骨科",
        "symptoms": ["颈部疼痛", "颈部僵硬", "头晕", "头痛", "上肢麻木", "手无力", "恶心", "耳鸣", "视力模糊", "吞咽困难"],
        "advice": "纠正不良姿势，避免长时间低头，进行颈椎操锻炼，理疗。"
    },
    "过敏性鼻炎": {
        "dept": "耳鼻喉科",
        "symptoms": ["鼻痒", "阵发性喷嚏", "清水样鼻涕", "鼻塞", "眼痒", "嗅觉减退", "头痛", "流泪", "咽痛", "咳嗽"],
        "advice": "避免接触过敏原，使用抗过敏药物，鼻腔冲洗。"
    },
    "湿疹": {
        "dept": "皮肤科",
        "symptoms": ["皮肤瘙痒", "红斑", "丘疹", "水疱", "渗出", "结痂", "皮肤干燥", "脱屑", "色素沉着", "皮肤增厚"],
        "advice": "保持皮肤清洁保湿，避免刺激，使用抗过敏药膏。"
    },
    "支气管炎": {
        "dept": "呼吸内科",
        "symptoms": ["咳嗽", "咳痰", "喘息", "胸闷", "气促", "发烧", "乏力", "食欲不振", "睡眠困难", "咽痛"],
        "advice": "戒烟，止咳化痰，抗感染治疗，注意保暖。"
    },
    "肺炎": {
        "dept": "呼吸内科",
        "symptoms": ["发烧", "寒战", "咳嗽", "咳痰", "胸痛", "呼吸困难", "乏力", "食欲不振", "恶心", "呕吐"],
        "advice": "抗生素治疗，卧床休息，多喝水，吸氧（必要时）。"
    },
    "哮喘": {
        "dept": "呼吸内科",
        "symptoms": ["喘息", "气促", "胸闷", "咳嗽", "夜间加重", "呼气困难", "焦虑", "心率加快", "大汗", "乏力"],
        "advice": "远离过敏原，规范使用吸入剂，随身携带急救药物。"
    },
    "胃溃疡": {
        "dept": "消化内科",
        "symptoms": ["上腹痛", "饭后痛", "反酸", "烧心", "恶心", "呕吐", "食欲减退", "体重下降", "黑便", "嗳气"],
        "advice": "规律饮食，戒烟戒酒，根除幽门螺杆菌，抑酸治疗。"
    },
    "胆囊炎": {
        "dept": "肝胆外科",
        "symptoms": ["右上腹痛", "恶心", "呕吐", "发烧", "黄疸", "腹胀", "食欲不振", "右肩放射痛", "畏寒", "嗳气"],
        "advice": "低脂饮食，消炎利胆，必要时手术治疗。"
    },
    "尿路感染": {
        "dept": "泌尿外科",
        "symptoms": ["尿频", "尿急", "尿痛", "下腹痛", "血尿", "腰痛", "发烧", "寒战", "恶心", "尿液混浊"],
        "advice": "多喝水冲刷尿道，抗生素治疗，注意个人卫生。"
    },
    "肾结石": {
        "dept": "泌尿外科",
        "symptoms": ["剧烈腰痛", "血尿", "恶心", "呕吐", "尿频", "尿急", "尿痛", "烦躁不安", "腹胀", "冷汗"],
        "advice": "多喝水，跳跃运动，止痛解痉，必要时碎石或手术。"
    },
    "焦虑症": {
        "dept": "精神心理科",
        "symptoms": ["过度担心", "紧张", "不安", "心悸", "胸闷", "出汗", "失眠", "注意力不集中", "肌肉紧张", "易怒"],
        "advice": "心理咨询，认知行为疗法，必要时药物治疗，放松训练。"
    },
    "抑郁症": {
        "dept": "精神心理科",
        "symptoms": ["情绪低落", "兴趣丧失", "疲劳", "睡眠障碍", "食欲改变", "自责", "注意力下降", "绝望感", "思维迟缓", "自杀念头"],
        "advice": "及时就医，心理治疗结合药物治疗，家人陪伴支持。"
    },
    "贫血": {
        "dept": "血液科",
        "symptoms": ["面色苍白", "乏力", "头晕", "心悸", "气短", "耳鸣", "注意力不集中", "手脚发凉", "食欲不振", "失眠"],
        "advice": "查明原因，补充铁剂或维生素B12，均衡饮食。"
    }
}

def generate_combinations():
    rules = []
    generated_signatures = set()
    
    # Target: 500+ rules
    target_count = 500
    
    diseases = list(DISEASE_DATA.keys())
    
    while len(rules) < target_count:
        # Pick a random disease
        disease_name = random.choice(diseases)
        data = DISEASE_DATA[disease_name]
        pool = data["symptoms"]
        
        # Pick 5 random symptoms (if pool < 5, duplicate allowed or pick less)
        k = 5
        if len(pool) >= k:
            selected = random.sample(pool, k)
        else:
            # Should not happen with our data, but for safety
            selected = pool + random.choices(pool, k=k-len(pool))
            
        # Sort to make signature unique regardless of order
        selected.sort()
        signature = ",".join(selected)
        
        if signature not in generated_signatures:
            rules.append({
                "keywords": signature,
                "disease_name": disease_name,
                "department": data["dept"],
                "advice": data["advice"]
            })
            generated_signatures.add(signature)
            
    return rules

def init_rules():
    with app.app_context():
        # Create table if not exists
        db.create_all()
        
        if SymptomDiseaseRule.query.first():
            print("Rules already initialized. Clearing old rules...")
            SymptomDiseaseRule.query.delete()
            db.session.commit()

        print("Generating 500+ rules...")
        rules_data = generate_combinations()
        
        print(f"Generated {len(rules_data)} rules. Inserting into DB...")
        
        # Batch insert
        objects = []
        for r in rules_data:
            obj = SymptomDiseaseRule(
                keywords=r["keywords"],
                disease_name=r["disease_name"],
                department=r["department"],
                advice=r["advice"]
            )
            objects.append(obj)
        
        try:
            db.session.add_all(objects)
            db.session.commit()
            print(f"Successfully added {len(objects)} rules.")
        except Exception as e:
            db.session.rollback()
            print(f"Error adding rules: {e}")

if __name__ == '__main__':
    init_rules()
