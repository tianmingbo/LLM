from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Role:
    key: str
    name: str
    prompt: str


ROLES: Dict[str, Role] = {
    "assistant": Role(
        key="assistant",
        name="通用助手",
        prompt="你是一个严谨、简洁、实用的中文 AI 助手，优先给出可执行建议。",
    ),
    "teacher": Role(
        key="teacher",
        name="耐心老师",
        prompt="你是一名循序渐进的老师，用中文解释概念，给出步骤和例子。",
    ),
    "coder": Role(
        key="coder",
        name="编程教练",
        prompt="你是一名资深编程教练，回答要包含关键代码思路、边界条件和调试建议。",
    ),
    "translator": Role(
        key="translator",
        name="翻译专家",
        prompt="你是一名翻译专家，保留原意并解释关键术语，必要时给出多种译法。",
    ),
}


def list_roles() -> List[dict]:
    return [
        {
            "key": role.key,
            "name": role.name,
            "prompt": role.prompt,
        }
        for role in ROLES.values()
    ]
