"""知识库缺口推荐规则配置。"""

from typing import Dict, List


KB_RECOMMENDATION_RULES: List[Dict[str, object]] = [
    {
        "keywords": ["仲裁", "独任", "仲裁员", "仲裁委员会"],
        "recommendation": (
            "《中华人民共和国仲裁法》（重点补齐：仲裁协议有效要件、"
            "仲裁委员会选定明确性、独任仲裁员选任程序）"
        ),
    },
    {
        "keywords": ["诉讼", "管辖", "管辖权", "法院"],
        "recommendation": (
            "《中华人民共和国民事诉讼法》（重点补齐：协议管辖法定连接点、"
            "专属管辖及级别管辖限制）"
        ),
    },
    {
        "keywords": ["知识产权", "软件著作权", "专利", "商业秘密"],
        "recommendation": (
            "《中华人民共和国著作权法》及《反不正当竞争法》（重点补齐："
            "职务成果权属约定、商业秘密保护边界）"
        ),
    },
    {
        "keywords": ["税", "发票", "代缴"],
        "recommendation": (
            "国家税务总局《发票管理办法》及增值税专用发票开具与货款结算联动规则"
        ),
    },
    {
        "keywords": ["劳动", "竞业限制", "辞职", "社保"],
        "recommendation": (
            "《中华人民共和国劳动合同法》（重点补齐：竞业限制经济补偿标准、"
            "用人单位单方解除权）"
        ),
    },
]

DEFAULT_RECOMMENDATION_QUERY_LIMIT = 3
DEFAULT_RECOMMENDATION = "相关专业领域法规"
FALLBACK_RECOMMENDATION_TEMPLATE = (
    "针对关键词 [{terms}] 的专项部委规章、司法解释或企业合规制度细则"
)
