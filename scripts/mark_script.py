#!/usr/bin/env python3
"""The one way to turn a chapter into a stage-play script: one `edit` at a time.

Role of the model: a screenwriter adapting reading text into a stage-play script. It
only *flags* places through the single `edit` tool (speak / delete / replace); the MCP
server performs the mechanical change in code (drop quotes, remove a redundant
``名字：`` attribution, insert a comma at a breath point).

A one-to-many **character dictionary** (canonical name -> labels: formal name, forms of
address, nicknames) is extracted and maintained at the START of every chapter; the
marker always uses the canonical name. There are no passersby: minor people keep their
name and can be turned into passerby voices later, at the TTS stage.

    AUDIOBOOK_LLM_BASE_URL=http://127.0.0.1:8080/v1 AUDIOBOOK_LLM_MODEL=gemma-4-12b \
    AUDIOBOOK_LLM_PROFILE=gemma-4-12b python scripts/mark_script.py 11 --count 5 --mode seq
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

THINK = os.environ.get("AUDIOBOOK_THINK", "1").lower() not in ("0", "off", "false", "no")
THINK_TOKEN = "<|think|>"  # Gemma: prepend to the system prompt

from audiobook.llm import LLMClient, config_from_env  # noqa: E402
from audiobook.marks import (  # noqa: E402
    MARK_OPEN,
    MARK_SEP,
    parse_marks,
    render_diff_html,
    render_html,
    unmarked_quotes,
)
from audiobook.mcp import MCPClient  # noqa: E402
from audiobook.schema import Cast  # noqa: E402

SYSTEM = """你是把「阅读文本」改编成「舞台剧台本」的编剧：人物说出的台词要标出发言角色，旁白原样保留，最终交机器朗读。
**你只有一个工具 `edit`，每次只改一处（单条 edit）**。定位片段**最多 6 个字**（能唯一确定即可）；抄整句一定失败。
- 台词 → `edit(op="speak", text="“带引号的整段”", role="人物词典里的规范名")`（引号连同内容给它，工具自动去引号）。
- 不是台词（术语、称呼、拟声、纯标点“……”，或只是强调）→ `edit(op="delete", text="“带引号的词”")`：工具去引号留下词；引号里只有标点则整段删。
- 标点 → `edit(op="replace", find="几个字", replace="几个字")`：补句末标点；**在人物说话的“气口”（换气/停顿处）加逗号**断开长句（不加句号）。
role 一律写**人物词典里的规范名**（不要把称呼/绰号写进 role）。

示例（照做）：
- 原文 `瑞贝卡抓着法杖：“龙…龙…”` → `edit(op="speak", text="“龙…龙…”", role="瑞贝卡")`。
- `高文：“……”` 这种「人名：」由 speak **自动删掉**，你不用单独删。
- `“…”` 只是省略，不是台词 → `edit(op="delete", text="“…”")`。
- ❌ 反面：`edit(op="speak", text="高文：“…”")`——把「人名+冒号+引号」当台词是错的。
把本章每一处都处理完再结束。
"""

ROSTER_SYSTEM = """你在维护一部小说的「人物词典」。词典把每个角色（规范名）映射到若干标签：正式名、称呼、绰号、代称、姓氏等（同一人可有很多标签）。
给定「已有词典」和「本章正文」，找出本章用来指代人物的**新标签**，归入对应角色；若某人是词典里没有的新角色，就以它的规范名新建一项。
只输出 JSON 对象：键=角色规范名，值=本章新增的标签数组（不要重复已有标签、不要输出空数组）。没有任何新增就输出 {}。
"""

COMPRESS_SYSTEM = """你在为长篇小说的台本标注任务维护「前情摘要」（相当于把一整条对话压缩成一段）。
给定「已有摘要」和「本章正文/出场角色」，输出更新后的滚动摘要：一段中文，尽量短（≤400 字）。
保留对后续判断「谁在说话」有用的信息：新出场人物的身份、称呼、绰号、人物关系、称呼变化、剧情要点。不要复述全文，不要漏掉新人物。
"""

TOOLS = ["edit"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("chapter", type=int, help="first chapter id")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--book", default="dawn")
    parser.add_argument(
        "--mode",
        choices=["seq", "sep"],
        default="seq",
        help="seq=reuse one annotation context across chapters; sep=fresh per chapter",
    )
    parser.add_argument("--max-steps", type=int, default=200, help="tool steps per chapter")
    return parser.parse_args()


def load_roster(book: str) -> dict[str, list[str]]:
    """Seed the dictionary from the fixed cast (canonical -> labels)."""
    path = APP_ROOT / "outputs" / book / "cast.json"
    if not path.is_file():
        return {}
    cast = Cast.load(str(path))
    roster: dict[str, list[str]] = {}
    for role in cast.roles.values():
        if role.kind == "narrator":
            continue
        roster[role.name] = [label for label in [role.name, *role.aliases] if label]
    return roster


def dict_text(roster: dict[str, list[str]], chapter_text: str, always: list[str]) -> str:
    """Inline the dictionary, keeping every label that appears in this chapter (+ mains)."""
    lines = []
    for name, labels in roster.items():
        if name in always or any(label and label in chapter_text for label in labels):
            lines.append(f"{name}（{'、'.join(dict.fromkeys(labels))}）")
    return "人物词典（规范名（标签…），role 只能写规范名）：\n- " + "\n- ".join(lines)


def maintain_roster(llm: LLMClient, roster: dict[str, list[str]], chapter_text: str) -> None:
    """Extract this chapter's new labels and merge them into the dictionary (in place)."""
    existing = "\n".join(f"{name}: {'、'.join(labels)}" for name, labels in roster.items())
    try:
        result = llm.chat_json(
            ROSTER_SYSTEM,
            f"【已有词典】\n{existing}\n\n【本章正文】\n{chapter_text}",
            thinking=THINK,
        )
    except Exception as error:  # noqa: BLE001 - dictionary is best-effort
        print(f"  [roster] 维护失败：{str(error)[:120]}", flush=True)
        return
    if not isinstance(result, dict):
        return
    added = 0
    for name, labels in result.items():
        name = str(name).strip()
        if not name:
            continue
        bucket = roster.setdefault(name, [name])
        for label in labels if isinstance(labels, list) else []:
            label = str(label).strip()
            if label and label not in bucket:
                bucket.append(label)
                added += 1
    if added:
        print(f"  [roster] 新增 {added} 个标签，词条 {len(roster)}", flush=True)


def compress(llm: LLMClient, summary: str, chapter_text: str, marked: str) -> str:
    """Archive this chapter and fold it into a short rolling summary (context compression)."""
    roles = sorted({seg["role_name"] for seg in parse_marks(marked) if seg["kind"] == "speech"})
    try:
        return llm.chat(
            [
                {"role": "system", "content": COMPRESS_SYSTEM},
                {
                    "role": "user",
                    "content": f"【已有摘要】\n{summary or '（无）'}\n\n【本章出场角色】{'、'.join(roles)}\n\n"
                    f"【本章正文】\n{chapter_text[-4000:]}",
                },
            ],
            max_tokens=700,
            thinking=False,
        ).strip()
    except Exception as error:  # noqa: BLE001 - summary is best-effort
        print(f"  [compress] 失败：{str(error)[:120]}", flush=True)
        return summary


def resolve_label(roster: dict[str, list[str]], name: str) -> str:
    """Map a written label/name to its canonical entry (or register it as new)."""
    if name in roster:
        return name
    for canonical, labels in roster.items():
        if name in labels:
            return canonical
    roster[name] = [name]
    return name


def run_turn(client, config, tools, mcp, messages, max_steps, counters) -> None:
    """Tool loop until the model stops calling tools. Guards against a failing retry loop."""
    errors = 0
    last_sig, repeats, fail_total = "", 0, 0
    for _step in range(1, max_steps + 1):
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                tools=tools or None,
                tool_choice="auto" if tools else None,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=4096,
                extra_body={"top_k": config.top_k, "chat_template_kwargs": {"enable_thinking": THINK}},
            )
        except Exception as error:  # noqa: BLE001 - malformed tool-call JSON -> retry
            errors += 1
            print(f"  [llm] error {errors}: {str(error)[:140]}", flush=True)
            if errors > 6:
                return
            messages.append({"role": "user", "content": "上一条工具调用参数 JSON 非法；请一次只改一处后重试。"})
            continue
        counters["llm_s"] += time.perf_counter() - started
        counters["llm_calls"] += 1
        message = response.choices[0].message
        tool_calls = [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in (message.tool_calls or [])
        ]
        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": tool_calls or None})
        if not tool_calls:
            return
        for call in tool_calls:
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            sig = call["function"]["name"] + call["function"]["arguments"]
            result = mcp.call(call["function"]["name"], arguments)
            counters["tool_calls"] += 1
            print(f"  {call['function']['name']} {str(arguments)[:60]} -> {result[:70]}", flush=True)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            if '"ok": false' in result or "not found" in result:
                fail_total += 1
                repeats = repeats + 1 if sig == last_sig else 0
                last_sig = sig
                left = unmarked_quotes(json.loads(mcp.call("get_marked", {}))["text"])
                messages.append(
                    {
                        "role": "user",
                        "content": "找不到目标。当前还没处理的引号是：\n"
                        + "\n".join(left[:40])
                        + "\n\n只从这些里挑一处，用 ≤6 个字的片段重试；没有就结束。",
                    }
                )
                if fail_total >= 5:
                    return


def main() -> None:
    args = parse_args()
    config = config_from_env()
    if config is None:
        raise SystemExit("no LLM configured")
    from openai import OpenAI

    client = OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=900)
    llm = LLMClient(config)
    mcp = MCPClient()
    mcp.initialize()
    all_tools = {t["function"]["name"]: t for t in mcp.openai_tools()}
    tools = [all_tools[name] for name in TOOLS]

    chapters = APP_ROOT / "outputs" / args.book / "chapters"
    out_dir = APP_ROOT / "outputs" / args.book / "script"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [cid for cid in range(args.chapter, args.chapter + args.count) if (chapters / f"ch{cid:03d}.txt").is_file()]

    roster = load_roster(args.book)
    always = list(roster)
    counters = {"llm_s": 0.0, "llm_calls": 0, "tool_calls": 0}
    started = time.perf_counter()
    phases: list[dict] = []
    summary = ""  # seq: one rolling conversation, compressed after every chapter

    for cid in ids:
        text = (chapters / f"ch{cid:03d}.txt").read_text(encoding="utf-8")
        maintain_roster(llm, roster, text)  # mine characters when the big text is injected
        system = (THINK_TOKEN if THINK else "") + SYSTEM + "\n\n" + dict_text(roster, text, always)
        preface = f"【前情摘要】\n{summary}\n\n" if (args.mode == "seq" and summary) else ""
        mcp.call("set_text", {"text": text})
        run_turn(
            client,
            config,
            tools,
            mcp,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{preface}处理第 {cid} 章：\n{text}"},
            ],
            args.max_steps,
            counters,
        )
        for _ in range(3):  # completeness: re-feed leftover quotes until none remain
            left = unmarked_quotes(json.loads(mcp.call("get_marked", {}))["text"])
            if not left:
                break
            run_turn(
                client,
                config,
                tools,
                mcp,
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": "还有这些引号没处理（是台词就 edit(op=speak)，否则 edit(op=delete)，逐处调用）：\n"
                        + "\n".join(left[:60]),
                    },
                ],
                args.max_steps,
                counters,
            )
        snapshot = json.loads(mcp.call("get_marked", {}))["text"]
        for seg in parse_marks(snapshot):  # normalise any label/alias to the canonical name
            if seg["kind"] != "speech":
                continue
            canonical = resolve_label(roster, seg["role_name"])
            if canonical != seg["role_name"]:
                snapshot = snapshot.replace(f"{MARK_OPEN}{seg['role_name']}{MARK_SEP}", f"{MARK_OPEN}{canonical}{MARK_SEP}")
        (out_dir / f"ch{cid:03d}.marked.txt").write_text(snapshot, encoding="utf-8")
        if args.mode == "seq":  # archive + compress the conversation before the next chapter
            summary = compress(llm, summary, text, snapshot)
        left = len(unmarked_quotes(snapshot))
        phases.append({"chapter": cid, "chars": len(snapshot), "leftover": left, "roles": len(roster)})
        print(f"[{args.mode}] ch{cid:03d} chars={len(snapshot)} leftover={left} 词条={len(roster)}", flush=True)
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")

    marked = "\n".join((out_dir / f"ch{cid:03d}.marked.txt").read_text(encoding="utf-8") for cid in ids)
    print(f"[check] 未标记的引号 {len(unmarked_quotes(marked))} 处", flush=True)
    (out_dir / "roles.json").write_text(json.dumps(roster, ensure_ascii=False, indent=2), encoding="utf-8")

    stem = f"ch{ids[0]:03d}_x{len(ids)}"
    render_html(parse_marks(marked), marked, out_dir / f"{stem}.html")
    original = "\n".join((chapters / f"ch{cid:03d}.txt").read_text(encoding="utf-8") for cid in ids)
    render_diff_html(original, marked, out_dir / f"{stem}.diff.html")

    timing = {
        "start": args.chapter,
        "count": len(ids),
        "mode": args.mode,
        "wall_s": round(time.perf_counter() - started, 1),
        "llm_s": round(counters["llm_s"], 1),
        "llm_calls": counters["llm_calls"],
        "tool_calls": counters["tool_calls"],
        "roles": len(roster),
        "phases": phases,
    }
    print(f"\n[timing] {json.dumps(timing, ensure_ascii=False)}")
    print(f"[html] {out_dir / f'{stem}.html'}")


if __name__ == "__main__":
    main()
