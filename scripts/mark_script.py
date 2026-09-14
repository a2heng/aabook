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
    live_fragments,
    parse_marks,
    render_diff_html,
    render_html,
    unmarked_quotes,
)
from audiobook.live import Live  # noqa: E402
from audiobook.mcp import MCPClient  # noqa: E402

SYSTEM = """你是把「小说」改编成「有声剧台本」的编剧。目标：听众闭着眼也能听明白——**谁在说话、说了什么、怎么停顿**。
程序用 ⦃角色名␟台词⦄ 表示台词，标记外一律旁白。你只指出"哪一处怎么改"，去引号/加逗号等机械动作由程序完成。

## 一、工具纪律
- 只有一个工具 `edit`，**一次一处**；改完接着下一处；**禁止一次多条、禁止输出正文**。
- 定位片段从正文**逐字复制**、**≤6 字**、唯一即可；抄整句必然 "not found"。
- 通读全章、心里先分清"谁说了哪句"，再逐处处理；改过的地方不要再动。
- **倒着改（重要）**：从本章**最后一处**开始，往**前**推进（章末 → 章首）。这样前面的改动不会挪动/破坏后面要处理的文字。
- 定位片段**不要跨越已经标好的 ⦃…⦄**；先做去引号/标台词，最后再统一加逗号。

## 二、台词怎么标（关键）
- 人物**说出**的话（引号内）→ `edit(op="speak", text="“引号连内容”", role="规范名")`：
  **只去掉引号**；引号前后的字（包括「人名：」「他说」）**全部保留，绝不删除**。
- **长台词**：按句拆成**多次 speak**（同一 role），每次给一小段（≤20 字、逐字复制）；程序会把相邻同角色片段**自动拼成一条**。
- 台词被旁白/动作打断：`“A，”他顿了顿，“B。”` → 分两次 speak（同一 role），中间旁白**保留不动**。
- 「人名：」只有**光秃秃的人名+冒号**（如「高文：」「赫蒂：」）→ 工具**自动删掉**；若带动作/神态/心理（「赫蒂抬起头说道：」「他心想：」）→ **保留不动**。

## 三、不是台词的引号 → delete（**只去引号，不删字**）
术语、绰号、强调、拟声（“铮”“轰”）→ 去引号、留词；引号里只有标点（“……”“——”“！”）→ 才整段删。
**除了引号/纯标点，任何字都不许删**（旁白、动作、神态一律保留）。

## 四、怎么判断说话人
- 找紧邻提示语 `X说/道/问/答/喊/笑道/低声道/自言自语…`（引号前后都算）。
- 对话一来一往；问话/应答之后的引号多半是对方。
- 「姑妈」「先祖大人」是**称呼不是说话人**，换成词典规范名。
- role 只能是词典**规范名**：不能是描述（"混血精灵"），不能带标点/冒号/人称。
- 拿不准就选词典里最可能的人，**绝不新造名字**。

## 五、让句子更短、更好听（用 replace，重点）
- **气口加逗号**：人物说话太长时，在换气/停顿处补 `，`，把长句断开（要多补）。
- 台词里表停顿/拖音的连续省略号 `……` → 换成 `，`；结巴式单个 `…`（“我…我…”）保留。
- 多个叹号/问号 `！！！`/`？？？` → 只留一个 `！`/`？`；破折号 `——`（打断）→ 换成 `，`。
- **不动**句末标点有无、不改词、不重写句子；引号外独行的纯标点（“……”“——”单独成段）→ 删掉。

## 六、正反例
✓ `赫蒂说道：“先祖大人，您回来了。”` → speak(role=赫蒂)，`赫蒂说道：` 保留为旁白
✓ `“我不同意，”他攥紧拳头，“但我服从。”` → 两次 speak（同一人），`他攥紧拳头，` 保留
✓ 长台词分多次 speak，程序自动拼合
✓ `“你……别过来！！！”` → replace `……`→`，`、`！！！`→`！`
✗ 删掉旁白/动作（`delete(text="他攥紧拳头")`）——禁止
✗ `edit(op="speak", role="混血精灵")`（role 是描述，不是规范名）
✗ `edit(op="speak", find="他还记得城门口发生过的所有事情")`（定位太长，必失败）
"""

ROSTER_SYSTEM = """你在维护一部小说的「人物词典」：**规范名 → 若干标签**（正式名、简称、称呼、绰号、头衔、代称）。
输入是「已有词典」和「本章正文」。请把本章出现、**指向人物**的称呼，归并到同一个人名下补进词典。

规则：
- 规范名取这个人**最完整/最正式**的称呼（如"高文·塞西尔"）；简称/昵称/称呼/头衔都进标签。
- 只收**指人**的称呼；地名、组织、物品、种族、群体（如"暗影界""塞西尔家族""巨龙""人群""卫兵""一个声音"）**不收**。
- 同一个人的多个称呼必须并入**同一条**，不要重复建条目。
- 已有条目**只增补标签**，不改名、不删除。
- 没有任何新增就输出 {}。

只输出 JSON：{"规范名": ["新标签", ...]}。
示例：{"高文·塞西尔": ["老祖宗","先祖大人"], "赫蒂": ["姑妈"]}
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
    parser.add_argument("--force", action="store_true", help="redo chapters that already have a marked file")
    return parser.parse_args()


def load_roster(book: str) -> dict[str, list[str]]:
    """Resume the one-to-many dictionary from ``script/roles.json`` (empty on a cold start).

    No cast is preloaded: the dictionary is grown incrementally, chapter by chapter, so the
    model can only use names it actually met in the text.
    """
    path = APP_ROOT / "outputs" / book / "script" / "roles.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {name: [label for label in labels if label] for name, labels in data.items()}


def dict_text(roster: dict[str, list[str]], chapter_text: str) -> str:
    """Inline only the dictionary entries whose labels appear in this chapter."""
    lines = []
    for name, labels in roster.items():
        if any(label and label in chapter_text for label in labels):
            lines.append(f"{name}（{'、'.join(dict.fromkeys(labels))}）")
    return "人物词典（规范名（标签…），role 只能写规范名）：\n- " + "\n- ".join(lines)


def maintain_roster(llm: LLMClient, roster: dict[str, list[str]], chapter_text: str) -> int:
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
        return 0
    if not isinstance(result, dict):
        return 0
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
    return added


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


def run_turn(
    client, config, tools, mcp, messages, max_steps, counters, live: Live | None = None, chapter: int = 0, original: str = ""
) -> None:
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
        if live is not None and message.content:
            live.emit("assistant", chapter=chapter, content=message.content[:2000])
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
            if live is not None:
                live.emit("tool", chapter=chapter, call=call["function"]["name"], args=arguments)
                live.emit(
                    "result", chapter=chapter, ok=('"ok": false' not in result and "not found" not in result), result=result[:500]
                )
                current = json.loads(mcp.call("get_marked", {}))["text"]
                live.set_state(chapter, len(current), live_fragments(original, current))
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
    if args.count > 0:
        ids = [cid for cid in range(args.chapter, args.chapter + args.count) if (chapters / f"ch{cid:03d}.txt").is_file()]
    else:  # count<=0 -> every chapter from `chapter` to the end
        ids = [cid for cid in sorted(int(p.stem[2:]) for p in chapters.glob("ch*.txt")) if cid >= args.chapter]

    live = Live(out_dir / "live.jsonl")
    live.emit("start", total=len(ids), book=args.book, mode=args.mode)
    roster = load_roster(args.book)
    counters = {"llm_s": 0.0, "llm_calls": 0, "tool_calls": 0}
    started = time.perf_counter()
    phases: list[dict] = []
    summary = ""  # seq: one rolling conversation, compressed after every chapter

    for cid in ids:
        target = out_dir / f"ch{cid:03d}.marked.txt"
        if target.is_file() and not args.force:  # resumable: skip chapters already marked
            if not (out_dir / f"ch{cid:03d}.json").is_file():  # backfill canvas state for the picker
                done = target.read_text(encoding="utf-8")
                live.set_state(
                    cid, len(done), live_fragments((chapters / f"ch{cid:03d}.txt").read_text(encoding="utf-8"), done), done=True
                )
            print(f"[skip] ch{cid:03d} 已存在", flush=True)
            continue
        text = (chapters / f"ch{cid:03d}.txt").read_text(encoding="utf-8")
        live.emit("chapter", chapter=cid, chars=len(text))
        live.set_state(cid, len(text), live_fragments(text, text))
        added = maintain_roster(llm, roster, text)  # mine characters when the big text is injected
        live.emit("roster", chapter=cid, added=added, roles=len(roster))
        system = (THINK_TOKEN if THINK else "") + SYSTEM + "\n\n" + dict_text(roster, text)
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
            live,
            cid,
            text,
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
                live,
                cid,
                text,
            )
        snapshot = json.loads(mcp.call("get_marked", {}))["text"]
        for seg in parse_marks(snapshot):  # normalise any label/alias to the canonical name
            if seg["kind"] != "speech":
                continue
            canonical = resolve_label(roster, seg["role_name"])
            if canonical != seg["role_name"]:
                snapshot = snapshot.replace(f"{MARK_OPEN}{seg['role_name']}{MARK_SEP}", f"{MARK_OPEN}{canonical}{MARK_SEP}")
        (out_dir / f"ch{cid:03d}.marked.txt").write_text(snapshot, encoding="utf-8")
        (out_dir / "roles.json").write_text(json.dumps(roster, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.mode == "seq":  # archive + compress the conversation before the next chapter
            summary = compress(llm, summary, text, snapshot)
        left = len(unmarked_quotes(snapshot))
        phases.append({"chapter": cid, "chars": len(snapshot), "leftover": left, "roles": len(roster)})
        print(f"[{args.mode}] ch{cid:03d} chars={len(snapshot)} leftover={left} 词条={len(roster)}", flush=True)
        live.set_state(cid, len(snapshot), live_fragments(text, snapshot), done=True)
        live.emit("done", chapter=cid, leftover=left, roles=len(roster), summary=summary)
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
