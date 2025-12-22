from __future__ import annotations
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class DiseaseRecord(BaseModel):
    disease_name: str = Field(..., alias="疾病名称")
    disease_code: str = Field(..., alias="疾病编码")
    main_symptoms: List[str] = Field(default_factory=list, alias="主要症状")
    symptom_description: str = Field(..., alias="症状描述")
    symptom_type: str = Field(..., alias="症状类型")
    probability_weight: float = Field(..., ge=0.0, le=1.0, alias="概率权重")
    urgency_level: str = Field(..., alias="紧急程度")
    source_url: str = Field(..., alias="来源链接")
    crawl_timestamp: datetime = Field(default_factory=datetime.utcnow, alias="爬取时间戳")

    @validator("disease_name", "disease_code", "symptom_description", "symptom_type", "urgency_level", "source_url")
    def not_empty(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("字段不能为空")
        return v.strip()

    @validator("main_symptoms")
    def symptoms_non_empty(cls, v: List[str]) -> List[str]:
        return [s.strip() for s in v if s and s.strip()]

    def to_chinese_json(self) -> Dict[str, Any]:
        return {
            "疾病名称": self.disease_name,
            "疾病编码": self.disease_code,
            "主要症状": self.main_symptoms,
            "症状描述": self.symptom_description,
            "症状类型": self.symptom_type,
            "概率权重": self.probability_weight,
            "紧急程度": self.urgency_level,
            "来源链接": self.source_url,
            "爬取时间戳": self.crawl_timestamp.isoformat(),
        }

