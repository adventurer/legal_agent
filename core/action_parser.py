"""ReAct Action 行解析。"""

import re
from typing import Optional, Tuple


def extract_action(text: str) -> Tuple[Optional[str], Optional[str]]:
    """解析单行 Action，兼容半角/全角括号和带引号参数。"""
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("Action:"):
            continue
        content = line[len("Action:"):].strip()
        match = re.fullmatch(r"([A-Za-z_][\w-]*)\s*[\(（](.*)[\)）]", content)
        if match:
            name, argument = match.groups()
            return name.strip(), argument.strip().strip("'\"“”‘’")
    return None, None
