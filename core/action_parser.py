"""ReAct Action 行解析。"""

import re
from typing import List, Optional, Tuple


ACTION_PATTERN = re.compile(r"([A-Za-z_][\w-]*)\s*[\(（](.*)[\)）]")


def extract_actions(text: str) -> List[Tuple[str, str]]:
    """提取一段模型输出中的全部 Action，兼容 Action 与调用分行。"""
    actions: List[Tuple[str, str]] = []
    waiting_for_call = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "Action:":
            waiting_for_call = True
            continue
        if line.startswith("Action:"):
            content = line[len("Action:"):].strip()
            waiting_for_call = not bool(content)
        elif not waiting_for_call:
            continue
        else:
            content = line
            waiting_for_call = False
        if not content:
            continue
        match = ACTION_PATTERN.fullmatch(content)
        if match:
            name, argument = match.groups()
            actions.append((name.strip(), argument.strip().strip("'\"“”‘’")))
    return actions


def extract_action(text: str) -> Tuple[Optional[str], Optional[str]]:
    """解析单行 Action，兼容半角/全角括号和带引号参数。"""
    actions = extract_actions(text)
    if actions:
        return actions[0]
    return None, None
