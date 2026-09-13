"""Checks for LLM pointer-based segmentation + declaration rewrite (no network)."""

from __future__ import annotations

import unittest

from audiobook.extract import Unit
from audiobook.schema import Cast, Role
from audiobook.segment import _segment_spans, segment_units


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def chat_json(self, system, user, thinking=None, **kwargs):
        return self.payload


def make_cast() -> Cast:
    cast = Cast()
    cast.add(Role(role_id="gaowen", name="高文"))
    cast.narrator()
    return cast


def make_unit(text: str, kind: str = "narration", role_id: str = "narrator", role_name: str = "旁白") -> Unit:
    return Unit(kind=kind, role_id=role_id, role_name=role_name, raw_text=text, tts_text=text)


def seg(units, client, **kwargs):
    return segment_units(units, make_cast(), client, **kwargs)  # type: ignore[arg-type]


def payload(*items):
    segments = []
    for item in items:
        text, pointer = item[0], item[1]
        speech = item[2] if len(item) > 2 else ""
        segments.append({"text": text, "next": pointer, "speech": speech})
    return {"segments": segments}


LONG = "这是一句很长的话。它没有标点符号。所以需要被模型切开。"


class SegmentSpansTest(unittest.TestCase):
    def test_exact_split(self):
        parts = ["这是一句很长的话", "它没有标点符号", "所以需要被模型切开"]
        text = "".join(parts)
        spans = _segment_spans(text, [{"text": part} for part in parts])
        self.assertIsNotNone(spans)
        assert spans is not None
        self.assertEqual([text[a:b] for a, b in spans], parts)

    def test_rewrite_rejected(self):
        self.assertIsNone(_segment_spans("原始文本", [{"text": "原始"}, {"text": "改写了"}]))


class PointerSegmentTest(unittest.TestCase):
    def test_pointer_boundaries(self):
        text = "第一句。第二句。第三句。"
        client = FakeClient(payload(("第一句。", "第二句"), ("第二句。", "第三句"), ("第三句。", "")))
        unit = make_unit(text, kind="dialogue", role_id="gaowen", role_name="高文")  # dialogue: no narration merge
        units = seg([unit], client, max_seconds=0.1)
        self.assertEqual([u.raw_text for u in units], ["第一句。", "第二句。", "第三句。"])
        self.assertEqual([u.tts_text for u in units], ["第一句。", "第二句。", "第三句。"])

    def test_adjacent_narration_merged(self):
        text = "第一句。第二句。"
        client = FakeClient(payload(("第一句。", "第二句"), ("第二句。", "")))
        units = seg([make_unit(text)], client, max_seconds=0.1)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].tts_text, text)

    def test_declaration_rewrite(self):
        text = "他低声说道：“你好。”"
        client = FakeClient(payload((text, "", "他低声自语：你好。")))
        units = seg([make_unit(text)], client, max_seconds=0.1)
        self.assertEqual(units[0].tts_text, "他低声自语：你好。")
        self.assertEqual(units[0].raw_text, text)
        self.assertIn("rewritten", units[0].flags)
        self.assertNotIn("source_mismatch", units[0].flags)

    def test_source_mismatch_flagged(self):
        text = "他低声说道。"
        client = FakeClient(payload(("完全不同的原文。", "", "")))
        units = seg([make_unit(text)], client, max_seconds=0.1)
        self.assertIn("source_mismatch", units[0].flags)

    def test_kind_and_role_inherited(self):
        unit = make_unit("你来。", kind="dialogue", role_id="gaowen", role_name="高文")
        client = FakeClient(payload(("你来。", "")))
        units = seg([unit], client, max_seconds=0.1)
        self.assertEqual(units[0].kind, "dialogue")
        self.assertEqual(units[0].role_id, "gaowen")

    def test_dialogue_edit_flagged(self):
        unit = make_unit("千万不要。", kind="dialogue", role_id="gaowen", role_name="高文")
        client = FakeClient(payload(("千万不要。", "", "千万别。")))
        units = seg([unit], client, max_seconds=0.1)
        self.assertIn("dialogue_edited", units[0].flags)

    def test_bad_anchor_flagged(self):
        text = "第一句。第二句。"
        client = FakeClient(payload(("第一句。", "完全找不到的锚"), ("第二句。", "")))
        units = seg([make_unit(text)], client, max_seconds=0.1)
        self.assertTrue(any("anchor_failed" in u.flags for u in units))

    def test_no_client_untouched(self):
        units = seg([make_unit(LONG)], None)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].raw_text, LONG)


if __name__ == "__main__":
    unittest.main()
