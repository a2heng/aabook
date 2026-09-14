"""Constrained resolver agent for ambiguous script rows.

The single-pass extractor labels a whole chapter at once; a handful of rows are
left ambiguous (typically dialogue attributed to the narrator). This agent
resolves only those rows. It may never rewrite text -- it can only assign roles
drawn from the cast.

Design note (empirical): a free-running multi-step tool loop was unreliable on a
9B local model (wrong picks, never reaching ``finish``). The model is much better
at a *single batched* decision with all evidence pre-gathered, so the
deterministic ``_context`` tool collects evidence and the model decides once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .extract import Unit, cast_block
from .llm import LLMClient, prompt_hash
from .schema import Cast

PROMPT_ID = "agent.resolve.v2"
AGENT_MAX_TOKENS = 1024

SYSTEM_TEMPLATE = """你是中文小说说话人判定器。给定若干"说话人未确定"的对白行及其上下文，
上下文每行格式为"角色/kind 原文"，目标行用 >> 标出。请判断每个目标行的说话人。
只输出 JSON：
{"fixes":[{"unit":<行号>,"role":"<角色名>"}]}

规则：
- role 必须来自角色表；无法确定（群像、多人、无归属）用"旁白"。
- 依据归属词（X说道/道/问/喊/答/补充道）、标点（冒号后引出的话）、句子主语和对话轮次判断。
- 对话与叙述交替时，注意"这句对白紧邻的叙述里是谁在说/谁在做动作"。
- 必须覆盖全部目标行，不要遗漏。

【角色表】
__CAST__
"""


@dataclass
class AgentResult:
    fixes: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    prompt_id: str = PROMPT_ID
    prompt_hash: str = ""


def build_system(cast: Cast) -> str:
    return SYSTEM_TEMPLATE.replace("__CAST__", cast_block(cast))


class RoleAgent:
    def __init__(self, cast: Cast, client: LLMClient, window: int = 4):
        self.cast = cast
        self.client = client
        self.window = window
        self.system = build_system(cast)
        self.system_hash = prompt_hash(PROMPT_ID, self.system)

    def targets(self, units: list[Unit]) -> list[int]:
        narrator = self.cast.narrator()
        return [
            index
            for index, unit in enumerate(units)
            if not {"agent_resolved", "role_repaired"} & set(unit.flags)
            and ((unit.kind == "dialogue" and unit.role_id == narrator.role_id) or "unresolved_role" in unit.flags)
        ]

    def _context(self, units: list[Unit], index: int) -> str:
        low = max(0, index - self.window)
        high = min(len(units), index + self.window + 1)
        lines = []
        for i in range(low, high):
            mark = ">>" if i == index else "  "
            lines.append(f"{mark} 行{i} [{units[i].role_name}/{units[i].kind}] {units[i].tts_text}")
        return "\n".join(lines)

    def _assign(self, units: list[Unit], index: int, role) -> None:
        units[index].role_id = role.role_id
        units[index].role_name = role.name
        units[index].confidence = max(units[index].confidence, 0.8)
        if "agent_resolved" not in units[index].flags:
            units[index].flags.append("agent_resolved")

    def run(self, units: list[Unit]) -> AgentResult:
        result = AgentResult(prompt_hash=self.system_hash)
        targets = self.targets(units)
        if not targets:
            return result
        blocks = [self._context(units, index) for index in targets]
        user = "目标行及上下文：\n\n" + "\n\n".join(blocks) + '\n\n只输出 {"fixes":[...]}，覆盖全部目标行。'
        payload = None
        for thinking in (True, False):
            try:
                payload = self.client.chat_json(self.system, user, max_tokens=AGENT_MAX_TOKENS, thinking=thinking)
                break
            except Exception:  # noqa: BLE001 - fall back to the other mode
                continue
        if payload is None:
            return result
        result.actions.append(payload if isinstance(payload, dict) else {"fixes": payload})
        items = payload.get("fixes") if isinstance(payload, dict) else payload
        target_set = set(targets)
        for item in items or []:
            if not isinstance(item, dict):
                continue
            index = _as_index(item.get("unit"))
            role = self.cast.resolve(str(item.get("role") or "").strip())
            if index is None or role is None or index not in target_set:
                continue
            self._assign(units, index, role)
            result.fixes.append({"unit": index, "role": role.name})
        return result


def _as_index(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
