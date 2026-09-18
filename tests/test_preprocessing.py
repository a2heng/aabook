"""Preprocessing + vocal-event checks (offline, no model/network)."""

from __future__ import annotations

import unittest

from audiobook.cleaning import normalize_text, split_chapters
from audiobook.marks import mark, parse_marks
from audiobook.tts import VOCAL_EVENTS, find_tags, is_event_tag, normalize_tag


class SpeechMergeTest(unittest.TestCase):
    def test_short_narration_between_same_role_speech_is_dropped_and_merged(self):
        text = f"{mark('甲', '第一句。')}他顿了顿。{mark('甲', '第二句。')}"
        segments = parse_marks(text)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["kind"], "speech")
        self.assertEqual(segments[0]["text"], "第一句。第二句。")

    def test_long_narration_keeps_speaker_change_boundary(self):
        middle = "他沉默了很久，回想起许多年前的往事，心情复杂。" * 2
        text = f"{mark('甲', '第一句。')}{middle}{mark('甲', '第二句。')}"
        kinds = [seg["kind"] for seg in parse_marks(text)]
        self.assertIn("narration", kinds)

    def test_adjacent_narration_merges(self):
        segments = parse_marks("第一段旁白。第二段旁白。")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "第一段旁白。第二段旁白。")


class BracketCleaningTest(unittest.TestCase):
    def test_square_bracket_symbols_are_removed_but_text_kept(self):
        text = "他笑了。[作者的话] 这是[笑]的测试。"
        self.assertEqual(normalize_text(text).strip(), "他笑了。作者的话 这是笑的测试。")

    def test_keeps_other_text(self):
        text = "【保留】这是“正常”的句子。"
        self.assertEqual(normalize_text(text).strip(), "【保留】这是“正常”的句子。")


class SiteBoilerplateTest(unittest.TestCase):
    def test_site_lines_and_decorations_are_removed(self):
        text = (
            "-----=====-----\n声明：本书由书荒部落收集整理，仅供试读\n更多精校小说请访问书荒部落(noveless.com) \n\n正文第一行。"
        )
        cleaned = normalize_text(text)
        self.assertNotIn("书荒部落", cleaned)
        self.assertNotIn("-----", cleaned)
        self.assertIn("正文第一行。", cleaned)

    def test_metadata_only_preface_is_dropped(self):
        text = "某书\n作者：某人\n内容简介：一句话。\n\n第1章 开始\n正文。"
        chapters = split_chapters(normalize_text(text))
        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0].title, "第1章 开始")

    def test_real_preface_is_kept(self):
        text = "写在前面\n这是一段真正的前言。\n\n第1章 开始\n正文。"
        chapters = split_chapters(normalize_text(text))
        self.assertEqual([c.title for c in chapters], ["前言", "第1章 开始"])


class VocalEventsTest(unittest.TestCase):
    def test_curated_set_is_categorized_and_nonempty(self):
        for tag in ("笑", "叹气", "咳嗽", "清嗓子", "抽泣", "打哈欠"):
            self.assertIn(tag, VOCAL_EVENTS)

    def test_free_form_short_cjk_tags_are_accepted(self):
        self.assertTrue(is_event_tag("无奈的冷笑"))
        self.assertFalse(is_event_tag(""))
        self.assertFalse(is_event_tag("laugh"))
        self.assertFalse(is_event_tag("[太长了这个标签不该通过]"))

    def test_normalize_and_find(self):
        self.assertEqual(normalize_tag(" [叹气] "), "叹气")
        self.assertEqual(find_tags("[叹气]好吧[冷笑]"), ["叹气", "冷笑"])
        self.assertEqual(find_tags("没有标签"), [])


class UnmarkedQuotesTest(unittest.TestCase):
    def test_long_quoted_span_is_detected(self):
        from audiobook.marks import unmarked_quotes

        long_span = "“" + "很长的台词内容" * 20 + "”"
        self.assertEqual(unmarked_quotes(long_span), [long_span])

    def test_quotes_inside_tags_are_not_pending(self):
        from audiobook.marks import mark, unmarked_quotes

        self.assertEqual(unmarked_quotes(mark("高文", "“算了。”")), [])

    def test_handled_quote_disappears_from_pending(self):
        from audiobook.marks import unmarked_quotes

        self.assertEqual(unmarked_quotes("他把固定视角当成口头禅。"), [])


class DeleteDanglingTest(unittest.TestCase):
    def test_empty_quote_swallows_dangling_attribution(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他说道：“”现场一片沉默。")
        result = server.edit(op="delete", text="他说道：")
        self.assertTrue(result["ok"])
        self.assertEqual(server.text, "现场一片沉默。")

    def test_punctuation_only_quote_swallows_attribution(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他叹道：“。”后面还有。")
        result = server.edit(op="delete", text="叹道：")
        self.assertTrue(result["ok"])
        self.assertEqual(server.text, "后面还有。")

    def test_quoted_empty_pair_is_tolerated(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他说道：“”现场一片沉默。")
        result = server.edit(op="delete", text="“”")
        self.assertTrue(result["ok"])
        self.assertEqual(server.text, "现场一片沉默。")

    def test_nonempty_speech_quote_is_still_guarded(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他说：“算了。”")
        result = server.edit(op="delete", text="算了。")
        self.assertFalse(result["ok"])
        self.assertEqual(server.text, "他说：“算了。”")

    def test_speak_accepts_text_without_quotes(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他说：你来了。")
        result = server.edit(op="speak", text="你来了。", role="高文")
        self.assertTrue(result["ok"])
        self.assertEqual(server.text, "他说：<高文>你来了。</高文>")


class PendingQuotesTest(unittest.TestCase):
    def test_punctuation_only_spans_are_ignored(self):
        from scripts.mark_script import pending_quotes

        self.assertEqual(pending_quotes("他说：“，”然后说：“好。”"), ["“好。”"])

    def test_empty_pair_is_ignored(self):
        from scripts.mark_script import pending_quotes

        self.assertEqual(pending_quotes("他说道：“”"), [])


class RosterMergeTest(unittest.TestCase):
    class _Stub:
        def __init__(self, payload):
            self.payload = payload

        def chat_json(self, *args, **kwargs):
            return self.payload

    def test_structural_keys_are_ignored_and_aliases_merge(self):
        from scripts.mark_script import maintain_roster

        roster = {"高文·塞西尔": ["高文·塞西尔", "高文", "老祖宗"]}
        profiles = {}
        payload = {
            "aliases": ["x"],
            "voice": {"age": "青年"},
            "高文": {
                "aliases": ["老祖宗", "开拓者"],
                "voice": {"age": "青年", "gender": "男", "sample": "我是高文·塞西尔。"},
            },
        }
        maintain_roster(RosterMergeTest._Stub(payload), roster, profiles, "text")
        self.assertNotIn("aliases", roster)
        self.assertNotIn("voice", roster)
        self.assertNotIn("高文", roster)
        self.assertIn("开拓者", roster["高文·塞西尔"])
        self.assertIn("高文·塞西尔", profiles)

    def test_unknown_name_opens_new_entry(self):
        from scripts.mark_script import maintain_roster

        roster = {"高文·塞西尔": ["高文·塞西尔", "高文"]}
        profiles = {}
        payload = {"贝蒂": {"aliases": ["小侍女"], "voice": {"sample": "我是贝蒂。"}}}
        maintain_roster(RosterMergeTest._Stub(payload), roster, profiles, "text")
        self.assertIn("贝蒂", roster)
        self.assertIn("贝蒂", profiles)

    def test_structural_keys_do_not_capture_existing_character(self):
        from scripts.mark_script import maintain_roster

        roster = {"琥珀": ["琥珀"]}
        payload = {
            "aliases": ["琥珀", "半精灵少女"],
            "voice": {"age": "青年"},
            "高文·塞西尔": {"aliases": ["高文"], "voice": {"sample": "我是高文·塞西尔。"}},
        }
        maintain_roster(RosterMergeTest._Stub(payload), roster, {}, "text")
        self.assertEqual(roster["琥珀"], ["琥珀"])  # structural payload must not touch existing entries
        self.assertIn("高文·塞西尔", roster)


class McpTagTest(unittest.TestCase):
    def test_speak_rejects_punctuation_only(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他说：“。”")
        result = server.edit(op="speak", text="“。”", role="张三")
        self.assertFalse(result["ok"])

    def test_speak_prepends_validated_tag(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他说：“算了。”")
        result = server.edit(op="speak", text="“算了。”", role="张三", tag="叹气")
        self.assertTrue(result["ok"])
        self.assertIn("<张三>[叹气]算了。</张三>", server.text)

    def test_speak_ignores_invalid_tag(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他说：“算了。”")
        result = server.edit(op="speak", text="“算了。”", role="张三", tag="laugh")
        self.assertTrue(result["ok"])
        self.assertIn("<张三>算了。</张三>", server.text)
        self.assertNotIn("[laugh]", server.text)


if __name__ == "__main__":
    unittest.main()
