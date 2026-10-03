#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: core/schemas.py
职责:
1. 定义合同审查系统的统一输入、输出与流转数据结构 (Pydantic v2)
2. 规范风险等级枚举 (RiskLevel) 与单个风险项 (ReviewItem)
3. 提供审查报告容器 (ContractReviewReport) 及其反序列化/容错清洗逻辑
4. 为后续 FastAPI 网关与 Streamlit 前端提供请求/响应数据契约
"""

from enum import Enum
from typing import List, Optional, Any, Dict, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class RiskLevel(str, Enum):
    """风险等级枚举"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    NOTICE = "Notice"

    @classmethod
    def _missing_(cls, value: object):
        """容错处理：当模型返回如 'high', 'HIGH', '中', '高' 时的清洗映射"""
        if isinstance(value, str):
            val_lower = value.strip().lower()
            if "high" in val_lower or "高" in val_lower:
                return cls.HIGH
            elif "med" in val_lower or "中" in val_lower:
                return cls.MEDIUM
            elif "low" in val_lower or "低" in val_lower:
                return cls.LOW
            elif "notice" in val_lower or "提示" in val_lower:
                return cls.NOTICE
        return cls.LOW


class ReviewItem(BaseModel):
    """单条合同条款的审查结果"""
    clause_topic: str = Field(
        ..., 
        description="审查的条款主题或违约点（如：乙方逾期交货的违约金比例）"
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.LOW,
        description="High、Medium、Low 或 Notice 四级，按共享风险分级规则判断"
    )
    risk_type: Optional[str] = Field(
        default=None,
        description="主要风险类型：法律合规、履约、商业或表述"
    )
    enterprise_risk_level: Optional[str] = Field(
        default=None,
        description="命中自编法典时记录原样的企业内部风险等级，与统一审查风险等级分开"
    )
    enterprise_basis: Optional[str] = Field(
        default=None,
        description="企业规则、企业知识库或通用资料的来源与定位，不作为法律依据"
    )
    confidence: Optional[str] = Field(
        default=None,
        description="结论置信度：高、中或低；独立于风险等级"
    )
    legal_effect: Optional[str] = Field(default=None, description="法律效力维度判断")
    commercial_impact: Optional[str] = Field(default=None, description="商业后果维度判断")
    remedy_cost: Optional[str] = Field(default=None, description="救济成本维度判断")
    affected_party: Optional[str] = Field(default=None, description="主要受影响方：甲方、乙方或双方")
    legal_basis: str = Field(
        ..., 
        description="支持法律结论的法规条文及其法规证据编号；无直接依据时应明确说明"
    )
    issue: str = Field(
        ..., 
        description="具体的法律或商业合规风险剖析"
    )
    suggested_revision: str = Field(
        ..., 
        description="针对该条款的具体修改或补充建议"
    )

    @field_validator("clause_topic", "legal_basis", "issue", "suggested_revision", mode="before")
    @classmethod
    def strip_whitespace(cls, v: Any) -> str:
        """去除首尾多余空白符"""
        if isinstance(v, str):
            return v.strip()
        return str(v) if v is not None else ""

    @field_validator("risk_type", mode="before")
    @classmethod
    def clear_risk_level_misclassified_as_type(cls, value: Any) -> Optional[str]:
        """Do not expose a risk grade as the separate risk category."""
        if not isinstance(value, str):
            return value
        normalized = value.strip().strip("* ")
        level_labels = {
            "高风险", "中风险", "低风险", "提示",
            "high", "medium", "med", "low", "notice",
        }
        if normalized.casefold() in {label.casefold() for label in level_labels}:
            return None
        return normalized or None


class ContractReviewReport(BaseModel):
    """最终审查报告容器"""
    reviews: List[ReviewItem] = Field(
        default_factory=list, 
        description="所有识别出的审查条款列表"
    )

    @property
    def high_risk_count(self) -> int:
        """高风险项统计"""
        return sum(1 for item in self.reviews if item.risk_level == RiskLevel.HIGH)

    @property
    def medium_risk_count(self) -> int:
        """中风险项统计"""
        return sum(1 for item in self.reviews if item.risk_level == RiskLevel.MEDIUM)

    @property
    def low_risk_count(self) -> int:
        """低风险项统计"""
        return sum(1 for item in self.reviews if item.risk_level == RiskLevel.LOW)

    @property
    def notice_count(self) -> int:
        """履约/商务提示项统计"""
        return sum(1 for item in self.reviews if item.risk_level == RiskLevel.NOTICE)


# ==================== 网关与调度层通信契约 ====================

class ReviewRequest(BaseModel):
    """API 审查请求结构体"""
    contract_text: str = Field(
        ..., 
        min_length=10, 
        description="待审查的合同原文文本"
    )
    max_turns: Optional[int] = Field(
        default=6, 
        ge=1, 
        le=30,
        description="ReAct 推理最大轮次限制"
    )
    stream: Optional[bool] = Field(
        default=True,
        description="是否启用打字机流式输出"
    )
    review_run_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=80,
        description="同一轮多条款审查共享的追踪批次 ID",
    )
    review_side: Literal["buyer", "seller", "neutral"] = Field(
        default="neutral",
        description="审查立场：甲方、乙方或中立",
    )


class ContractRevisionItem(BaseModel):
    """用户确认后用于重写的单个合同条款。"""
    index: int
    title: str
    original_text: str
    suggested_revision: str


class ContractRewriteRequest(BaseModel):
    """按用户选择的条款生成修订版合同。"""
    clauses: List[Dict[str, Any]] = Field(..., min_length=1)
    review_report: str = Field(..., min_length=1)
    selected_indices: List[int] = Field(default_factory=list)


class ContractRewriteResponse(BaseModel):
    """修订版合同及逐条修改结果。"""
    contract_text: str
    revised_clauses: List[Dict[str, Any]] = Field(default_factory=list)


class AgentExecutionResult(BaseModel):
    """核心 Agent Loop 执行完毕后的完整状态容器"""
    status: str = Field(
        ..., 
        description="执行状态: success / partial_success / timeout / error"
    )
    turns: int = Field(
        ..., 
        description="实际交互推理轮数"
    )
    report: Optional[ContractReviewReport] = Field(
        default=None, 
        description="解析成功的结构化审查报告对象"
    )
    raw_report: str = Field(
        default="", 
        description="模型输出的原始 Final 文本"
    )
    error_message: Optional[str] = Field(
        default=None, 
        description="执行过程中的异常报错信息（如有）"
    )


# ==================== 本地简单验证 ====================
if __name__ == "__main__":
    test_dict = {
        "reviews": [
            {
                "clause_topic": "违约金过高条款",
                "risk_level": "high",  # 测试小写容错
                "legal_basis": "《民事法规.pdf》第 12 页",
                "issue": "约定违约金为总金额 5%，严重高于同类标准上限",
                "suggested_revision": "修改为每日万分之五"
            }
        ]
    }
    
    report = ContractReviewReport.model_validate(test_dict)
    print("解析成功:")
    print(f"审查项数量: {len(report.reviews)}")
    print(f"风险项级别: {report.reviews[0].risk_level}")
    print(f"高风险统计: {report.high_risk_count}")
