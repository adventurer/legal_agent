"""Deterministic commercial-risk indicators for contract text."""

import re
from decimal import Decimal
from typing import List

from .contracts import GuardrailFinding


_UNLIMITED_LIABILITY_PATTERNS = (
    re.compile(r"不设(?:置)?(?:任何)?责任上限|不设最高(?:赔偿)?限额"),
    re.compile(r"(?:赔偿责任|违约责任|赔偿金额)[^。；;\n]{0,40}(?:无上限|不设(?:置)?(?:任何)?上限|不受(?:任何)?限制)"),
)
_DAILY_RATE_PATTERN = re.compile(
    r"(?:每日|每天|按日)[^。；;\n]{0,32}?([0-9]+(?:\.[0-9]+)?)\s*[％%]"
)


def detect_contract_flags(contract_text: str) -> List[GuardrailFinding]:
    findings: List[GuardrailFinding] = []

    for pattern in _UNLIMITED_LIABILITY_PATTERNS:
        match = pattern.search(contract_text)
        if match:
            findings.append(GuardrailFinding(
                code="unlimited_liability",
                severity="high",
                message=(
                    "程序规则发现可能没有责任上限的表述。请核对责任主体、适用损失和例外；"
                    "这是商业风险标记，不是违法或无效结论。"
                ),
                evidence=match.group(0),
            ))
            break

    rates = [
        (Decimal(match.group(1)), match.group(0))
        for match in _DAILY_RATE_PATTERN.finditer(contract_text)
        if Decimal(match.group(1)) > 0
    ]
    if len(rates) >= 2:
        lowest = min(rate for rate, _ in rates)
        highest = max(rate for rate, _ in rates)
        multiplier = highest / lowest
        if multiplier > 5:
            findings.append(GuardrailFinding(
                code="daily_rate_asymmetry",
                severity="medium",
                message=(
                    f"程序规则计算出日费率倍率约为 {multiplier:.2f} 倍（超过 5 倍）。"
                    "请再核对计算基数、触发条件和实际承担方；倍率本身不证明显失公平。"
                ),
                evidence="；".join(rate_text for _, rate_text in rates[:4]),
            ))
    return findings