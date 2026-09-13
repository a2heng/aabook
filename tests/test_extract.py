"""Pure-function checks for the plan-02 numbered extractor (no network)."""

from __future__ import annotations

import unittest

from audiobook.cleaning import Chapter
from audiobook.extract import Unit, _minimal_units, extract_chapter
from audiobook.schema import Cast, Role


class FakeClient:
    """Minimal ``LLMClient`` stand-in returning a fixed payload."""

    def __init__(self, payload):
        self.payload = payload
        self.calls: list[dict] = []

    def chat_json(self, system, user, thinking=None, **kwargs):
        self.calls.append({"system": system, "user": user, "thinking": thinking})
        return self.payload


def make_cast() -> Cast:
    cast = Cast()
    cast.add(Role(role_id="gaowen", name="高文"))
    cast.add(Role(role_id="hedi", name="赫蒂"))
    cast.narrator()
    return cast


def joined(units: list[Unit]) -> str:
    return "".join(unit.raw_text for unit in units)


class ExtractChapterTest(unittest.TestCase):
    def test_no_client_is_single_narration(self):
        chapter = Chapter(chapter_id=1, title="第一章", text="高文穿越了。\n他醒了。")
        units = extract_chapter(chapter, make_cast(), None)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].kind, "narration")
        self.assertEqual(units[0].role_id, "narrator")
        self.assertEqual(units[0].raw_text, chapter.text)

    def test_minimal_units_are_quote_aware(self):
        units = _minimal_units("高文说：“你好。”他走了。")
        self.assertEqual([unit["inside"] for unit in units], [False, True, False])

    def test_numbered_labels_map_to_units(self):
        chapter = Chapter(chapter_id=1, title="第一章", text="高文说：“你好。”")
        client = FakeClient({"speakers": {"2": "高文"}})
        units = extract_chapter(chapter, make_cast(), client)  # type: ignore[arg-type]
        self.assertEqual([unit.kind for unit in units], ["narration", "dialogue"])
        self.assertEqual(units[1].role_name, "高文")
        self.assertEqual(joined(units), chapter.text)

    def test_unquoted_never_inherits_speaker(self):
        chapter = Chapter(chapter_id=1, title="第一章", text="“你来了。”他坐下。")
        client = FakeClient({"speakers": {"1": "高文"}})
        units = extract_chapter(chapter, make_cast(), client)  # type: ignore[arg-type]
        self.assertEqual([unit.kind for unit in units], ["dialogue", "narration"])
        self.assertEqual(units[1].role_id, "narrator")

    def test_numbered_quote_without_speaker_is_flagged(self):
        chapter = Chapter(chapter_id=1, title="第一章", text="“谁在那儿？”")
        client = FakeClient({"speakers": {}})
        units = extract_chapter(chapter, make_cast(), client)  # type: ignore[arg-type]
        self.assertEqual(units[0].kind, "dialogue")
        self.assertIn("unresolved_role", units[0].flags)

    def test_quote_marked_narration_is_narration(self):
        chapter = Chapter(chapter_id=1, title="第一章", text="“第一王朝”的气息。")
        client = FakeClient({"speakers": {"1": "旁白"}})
        units = extract_chapter(chapter, make_cast(), client)  # type: ignore[arg-type]
        self.assertEqual([unit.kind for unit in units], ["narration"])

    def test_exclamation_marked_narration_stays_dialogue(self):
        chapter = Chapter(chapter_id=1, title="第一章", text="“祖先啊！”")
        client = FakeClient({"speakers": {"1": "旁白"}})
        units = extract_chapter(chapter, make_cast(), client)  # type: ignore[arg-type]
        self.assertEqual(units[0].kind, "dialogue")
        self.assertIn("unresolved_role", units[0].flags)


if __name__ == "__main__":
    unittest.main()
