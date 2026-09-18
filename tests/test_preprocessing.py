"""Preprocessing + vocal-event checks (offline, no model/network)."""

from __future__ import annotations

import unittest

from audiobook.cleaning import normalize_text, split_chapters
from audiobook.marks import mark, parse_marks
from audiobook.tts import VOCAL_EVENTS, find_tags, is_event_tag, normalize_tag


class SpeechMergeTest(unittest.TestCase):
    def test_short_narration_is_kept_by_default(self):
        text = f"{mark('甲', '第一句。')}他顿了顿。{mark('甲', '第二句。')}"
        segments = parse_marks(text)
        self.assertEqual([seg["kind"] for seg in segments], ["speech", "narration", "speech"])
        self.assertEqual(segments[1]["text"], "他顿了顿。")

    def test_short_narration_can_be_merged_when_enabled(self):
        import audiobook.marks as marks

        original = marks.MAX_INTERRUPT_CHARS
        marks.MAX_INTERRUPT_CHARS = 12
        try:
            text = f"{mark('甲', '第一句。')}他顿了顿。{mark('甲', '第二句。')}"
            segments = parse_marks(text)
        finally:
            marks.MAX_INTERRUPT_CHARS = original
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

    def test_empty_quote_swallows_attribution_within_10_char_window(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他沉默了很久，终于缓缓叹道：“”屋里没人接话。")
        result = server.edit(op="delete", text="叹道：")
        self.assertTrue(result["ok"])
        self.assertEqual(server.text, "他沉默了很久，屋里没人接话。")

    def test_symbol_only_quote_triggers_the_same_check(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他愣了一下，叹道：“ ， ”屋子里没人接话。")
        result = server.edit(op="delete", text="叹道：")
        self.assertTrue(result["ok"])
        self.assertEqual(server.text, "他愣了一下，屋子里没人接话。")

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


class AnchorEndTest(unittest.TestCase):
    def test_start_and_end_anchor_marks_the_whole_unquoted_line(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他说道：这件事要从很久以前说起，中间还有很多内容，最后我们终于赢了。")
        result = server.edit(op="speak", text="这件事要从很久以前说起", role="陈默", end="我们终于赢了。")
        self.assertTrue(result["ok"])
        self.assertTrue(result.get("anchored"))
        self.assertIn("<陈默>这件事要从很久以前说起，中间还有很多内容，最后我们终于赢了。</陈默>", server.text)

    def test_short_line_without_end_still_works(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他说道：“你来了。”")
        result = server.edit(op="speak", text="你来了。", role="陈默")
        self.assertTrue(result["ok"])
        self.assertIn("<陈默>“你来了。”</陈默>", server.text)

    def test_anchors_never_span_two_quotes(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("“我先走了。”他披上外套，又回头补了一句，“明天见。”")
        result = server.edit(op="speak", text="我先走了。", end="明天见。", role="陈默")
        self.assertTrue(result["ok"])
        self.assertIn("<陈默>“我先走了。”</陈默>", server.text)
        # the second quote must NOT be swallowed together with the narration
        self.assertNotIn("<陈默>“我先走了。”他披上外套", server.text)
        self.assertTrue(server.text.rstrip().endswith("“明天见。”"))

    def test_complete_text_never_glues_two_quotes(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("“我先走了。”他披上外套，又回头补了一句，“明天见。”")
        result = server.edit(op="speak", text="“我先走了。”他披上外套，又回头补了一句，“明天见。”", role="陈默")
        self.assertTrue(result["ok"])
        self.assertIn("<陈默>“我先走了。”</陈默>", server.text)
        self.assertNotIn("他披上外套，又回头补了一句，“明天见。”</陈默>", server.text)

    def test_pause_and_continue_marks_both_spans(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("“我先走了。”他披上外套，又回头补了一句，“明天见。”")
        server.edit(op="speak", text="我先走了。", role="陈默")
        server.edit(op="speak", text="明天见。", role="陈默")
        self.assertIn("<陈默>“我先走了。”</陈默>", server.text)
        self.assertIn("<陈默>“明天见。”</陈默>", server.text)
        self.assertIn("他披上外套", server.text)


class RosterMergeTest(unittest.TestCase):
    class _Stub:
        def __init__(self, payload):
            self.payload = payload

        def chat_json(self, *args, **kwargs):
            return self.payload

    def test_structural_keys_are_ignored_and_aliases_merge(self):
        from scripts.mark_script import maintain_roster

        roster = {"高文·塞西尔": ["高文·塞西尔", "高文", "老祖宗"]}
        payload = {
            "aliases": ["x"],
            "voice": {"age": "青年"},
            "高文": {"aliases": ["老祖宗", "开拓者"], "voice": {"age": "青年", "gender": "男"}},
        }
        maintain_roster(RosterMergeTest._Stub(payload), roster, "text")
        self.assertNotIn("aliases", roster)
        self.assertNotIn("voice", roster)
        self.assertNotIn("高文", roster)
        self.assertIn("开拓者", roster["高文·塞西尔"])

    def test_unknown_name_opens_new_entry(self):
        from scripts.mark_script import maintain_roster

        roster = {"高文·塞西尔": ["高文·塞西尔", "高文"]}
        payload = {"贝蒂": {"aliases": ["小侍女"]}}
        maintain_roster(RosterMergeTest._Stub(payload), roster, "text")
        self.assertIn("贝蒂", roster)

    def test_full_name_seen_later_is_promoted(self):
        from scripts.mark_script import maintain_roster

        roster = {"高文": ["高文", "老祖宗"]}
        payload = {"高文·塞西尔": {"aliases": ["高文"]}}
        maintain_roster(RosterMergeTest._Stub(payload), roster, "text")
        self.assertNotIn("高文", roster)
        self.assertEqual(roster["高文·塞西尔"][0], "高文·塞西尔")
        self.assertIn("高文", roster["高文·塞西尔"])
        self.assertIn("老祖宗", roster["高文·塞西尔"])

    def test_aliases_are_not_capped(self):
        from scripts.mark_script import maintain_roster

        roster = {}
        payload = {"陈默": {"aliases": [f"称呼{i}" for i in range(8)]}}
        maintain_roster(RosterMergeTest._Stub(payload), roster, "text")
        self.assertEqual(len(roster["陈默"]) - 1, 8)

    def test_structural_keys_do_not_capture_existing_character(self):
        from scripts.mark_script import maintain_roster

        roster = {"琥珀": ["琥珀"]}
        payload = {"aliases": ["琥珀", "半精灵少女"], "高文·塞西尔": {"aliases": ["高文"]}}
        maintain_roster(RosterMergeTest._Stub(payload), roster, "text")
        self.assertEqual(roster["琥珀"], ["琥珀"])  # structural payload must not touch existing entries
        self.assertIn("高文·塞西尔", roster)


class AnchorMarkingTest(unittest.TestCase):
    def test_speak_with_long_anchor_marks_whole_quote(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他叹道：“这是一段很长的台词，后面还有更多的内容。”")
        result = server.edit(op="speak", text="这是一段很长的", role="高文")
        self.assertTrue(result["ok"])
        self.assertEqual(server.text, "他叹道：<高文>“这是一段很长的台词，后面还有更多的内容。”</高文>")

    def test_delete_with_long_anchor_strips_whole_quote(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他把“固定视角永远改变了他的命运”当成口头禅。")
        result = server.edit(op="delete", text="固定视角永远")
        self.assertTrue(result["ok"])
        self.assertNotIn("“", server.text)
        self.assertIn("固定视角永远改变了他的命运", server.text)

    def test_speak_mid_quote_fragment_marks_whole_quote(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他叹道：“这是一段很长的台词，后面还有更多的内容。”")
        result = server.edit(op="speak", text="很长的台词", role="高文")
        self.assertTrue(result["ok"])
        self.assertTrue(result.get("expanded"))
        self.assertEqual(server.text, "他叹道：<高文>“这是一段很长的台词，后面还有更多的内容。”</高文>")

    def test_delete_mid_quote_fragment_strips_whole_quote(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他把“固定视角永远改变了他的命运”当成口头禅。")
        result = server.edit(op="delete", text="永远改变了")
        self.assertTrue(result["ok"])
        self.assertNotIn("“", server.text)
        self.assertIn("固定视角永远改变了他的命运", server.text)

    def test_speak_tolerates_inner_quote_marks(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他上前一步：“父王，您认为那位‘复活’的大公是真是假？”")
        result = server.edit(op="speak", text="父王，您认为那位复活的大公是真是假？", role="拜伦")
        self.assertTrue(result["ok"])
        self.assertIn("<拜伦>“父王，您认为那位‘复活’的大公是真是假？”</拜伦>", server.text)

    def test_speak_keeps_inner_quote_marks(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他说道：“那位古人给了我们一个大大的‘惊喜’。”")
        result = server.edit(op="speak", text="那位古人给了我们一个大大的", role="高文")
        self.assertTrue(result["ok"])
        self.assertIn("<高文>“那位古人给了我们一个大大的‘惊喜’。”</高文>", server.text)

    def test_delete_removes_single_quote_pair(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他把它称为‘惊喜’然后走了。")
        result = server.edit(op="delete", text="惊喜")
        self.assertTrue(result["ok"])
        self.assertEqual(server.text, "他把它称为惊喜然后走了。")

    def test_speak_tolerates_punctuation_changes(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他一把抓过旁边的人：“快，派个会变鸟的德鲁伊！去皇冠街四号，让他们速做准备！”")
        result = server.edit(op="speak", text="快，派个会变鸟的德鲁伊，去皇冠街四号，让他们速做准备", role="高文")
        self.assertTrue(result["ok"])
        self.assertTrue(result.get("expanded"))
        self.assertIn("<高文>“快，派个会变鸟的德鲁伊！去皇冠街四号，让他们速做准备！”</高文>", server.text)

    def test_speak_matches_ellipsis_punctuation_variant(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("“抱……抱歉……”这位女士慌张地道着歉。")
        result = server.edit(op="speak", text="抱，抱歉", role="埃德蒙")
        self.assertTrue(result["ok"])
        self.assertEqual(server.text, "<埃德蒙>“抱……抱歉……”</埃德蒙>这位女士慌张地道着歉。")

    def test_delete_matches_punctuation_variant_inside_quotes(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他被称为“万物之耻，第一人”然后走了。")
        result = server.edit(op="delete", text="万物之耻第一人")
        self.assertTrue(result["ok"])
        self.assertNotIn("“", server.text)
        self.assertIn("万物之耻，第一人", server.text)

    def test_speak_anchor_outside_quotes_marks_only_fragment(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他转身离开，随后又停下脚步。")
        result = server.edit(op="speak", text="随后又停下", role="高文")
        self.assertTrue(result["ok"])
        self.assertEqual(server.text, "他转身离开，<高文>随后又停下</高文>脚步。")


class RoleAttributionTest(unittest.TestCase):
    def test_different_attribution_corrects_role(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_name_index({"高文": "高文", "琥珀": "琥珀"})
        server.set_text("高文：“……我吃饱撑的跟你这个万物之耻讲道理！”")
        result = server.edit(op="speak", text="我吃饱撑的", role="琥珀")
        self.assertTrue(result["ok"])
        self.assertEqual(result["role"], "高文")
        self.assertIn("role_note", result)
        self.assertEqual(server.text, "<高文>“……我吃饱撑的跟你这个万物之耻讲道理！”</高文>")

    def test_same_attribution_is_dropped(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_name_index({"高文": "高文"})
        server.set_text("高文：“你来了。”")
        result = server.edit(op="speak", text="你来了", role="高文")
        self.assertTrue(result["ok"])
        self.assertEqual(server.text, "<高文>“你来了。”</高文>")


class LiveFragmentsTest(unittest.TestCase):
    def test_removed_quote_marks_do_not_split_speech(self):
        from audiobook.marks import live_fragments

        original = "他说：“大大的‘惊喜’。”"
        marked = "他说：<高文>大大的惊喜。</高文>"
        fragments = live_fragments(original, marked)
        speech = [frag["text"] for frag in fragments if frag["kind"] == "speech"]
        self.assertEqual(speech, ["大大的惊喜。"])


class SpeakValidationTest(unittest.TestCase):
    def test_speak_rejects_punctuation_only(self):
        from scripts.mark_script import ScriptServer

        server = ScriptServer()
        server.set_text("他说：“。”")
        result = server.edit(op="speak", text="“。”", role="张三")
        self.assertFalse(result["ok"])

    def test_existing_bracket_events_pass_through_parsing(self):
        from audiobook.marks import parse_marks
        from audiobook.tts import find_tags

        segments = parse_marks("<高文>[叹气]好吧。</高文>")
        self.assertEqual(find_tags(segments[0]["text"]), ["叹气"])


if __name__ == "__main__":
    unittest.main()
