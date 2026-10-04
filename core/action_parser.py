"""ReAct Action 行解析。"""

import re
from typing import Optional, Tuple


ACTION_PATTERN = re.compile(r"([A-Za-z_][\w-]*)\s*[\(（](.*)[\)）]")
MAX_ACTION_ARGUMENT_CHARS = 500


def extract_action(text: str) -> Tuple[Optional[str], Optional[str]]:
    """只接受单独的 Thought + Action 回复，拒绝正文中的伪 Action。"""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) < 2 or not lines[0].startswith("Thought:"):
        return None, None
    if not lines[0][len("Thought:"):].strip():
        return None, None

    if lines[1] == "Action:":
        if len(lines) != 3:
            return None, None
        action_line = lines[2]
    elif lines[1].startswith("Action:"):
        if len(lines) != 2:
            return None, None
        action_line = lines[1][len("Action:"):].strip()
    else:
        return None, None

    match = ACTION_PATTERN.fullmatch(action_line)
    if not match:
        return None, None
    name, argument = match.groups()
    argument = argument.strip().strip("'\"“”‘’")
    if not argument or len(argument) > MAX_ACTION_ARGUMENT_CHARS:
        return None, None
    return name.strip(), argument
