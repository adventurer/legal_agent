"""OpenAI-compatible schemas for the approved read-only review tools."""

from typing import Any, Callable, Dict, Iterable

from .contracts import FinalReportArguments, QueryArguments


MAX_ADVERTISED_TOOLS = 5
FINAL_REPORT_TOOL_NAME = "submit_final_report"

READ_ONLY_TOOL_DESCRIPTIONS = {
    "search_civil_code": (
        "检索与合同问题相关的现行法律法规依据。使用具体法律概念、义务或规则名称，"
        "例如“买卖合同 检验期限”；避免只搜“中国 法规 合同法”等泛词。"
    ),
    "get_company_policy": "检索企业合规资料和企业内部审查规则。",
    "get_past_review_rules": (
        "检索企业自编法典及历史审查规则。命中规则后，应按其风险等级、审查标准和禁止情形"
        "审查相关合同条款，并参考建议条款形成企业内部风险判断和谈判建议；不得将企业规则作为法律依据。"
    ),
    "search_general_materials": "检索与合同问题相关的通用业务资料。",
}


def build_tool_schemas(
    tool_mapping: Dict[str, Callable[..., Any]],
    unavailable_tools: Iterable[str] = (),
    guardrail_codes: Iterable[str] = (),
) -> list[dict[str, Any]]:
    unavailable = set(unavailable_tools)
    guardrails = list(guardrail_codes)
    parameters = QueryArguments.model_json_schema()
    schemas = []
    for name, description in READ_ONLY_TOOL_DESCRIPTIONS.items():
        if name not in tool_mapping or name in unavailable:
            continue
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        })
        if len(schemas) == MAX_ADVERTISED_TOOLS - 1:
            break
    final_parameters = FinalReportArguments.model_json_schema()
    guardrail_schema = final_parameters["properties"]["acknowledged_guardrails"]
    guardrail_schema["maxItems"] = len(guardrails)
    if guardrails:
        guardrail_schema["items"]["enum"] = guardrails
        final_parameters.setdefault("required", []).append("acknowledged_guardrails")
    guardrail_instruction = (
        f"本次允许的程序规则代码：{', '.join(guardrails)}。"
        if guardrails else
        "本次没有程序规则代码，acknowledged_guardrails 可省略（按空数组处理）或传空数组。"
    )
    schemas.append({
        "type": "function",
        "function": {
            "name": FINAL_REPORT_TOOL_NAME,
            "description": (
                "提交最终合同审查报告。report 是 JSON 对象，内部只包含 reviews；"
                "每条 review 都必须包含非空 suggested_revision，无需改约时也要明确说明；"
                "acknowledged_guardrails 是与 report 同级的顶层数组，不能放进 report 内；"
                "acknowledged_guardrails 只能包含程序规则代码，不得填写工具名、证据编号或法规编号；"
                f"{guardrail_instruction}"
            ),
            "parameters": final_parameters,
        },
    })
    return schemas