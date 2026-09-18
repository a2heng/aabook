"""Pure-function checks for the pre-LLM TTS character filter (no network)."""

from __future__ import annotations

import unittest

from audiobook.textnorm import clean_for_llm, filter_tts_chars, normalize_tts, one_paragraph


class FilterTtsCharsTest(unittest.TestCase):
    def test_drops_symbols_and_decorations(self):
        self.assertEqual(filter_tts_chars("★☆高文·塞西尔※※"), "高文·塞西尔")

    def test_keeps_letters_digits_and_punctuation(self):
        text = "High 3.14，是吗？“第一王朝”——Abbé!"
        self.assertEqual(filter_tts_chars(text), text)

    def test_drops_box_drawing_and_math(self):
        self.assertEqual(filter_tts_chars("√2 × 3 = 6￥ ○"), "2  3  6 ")

    def test_keeps_newlines_and_tabs(self):
        self.assertEqual(filter_tts_chars("a\nb\tc"), "a\nb\tc")

    def test_clean_for_llm_filters_then_normalises(self):
        self.assertEqual(clean_for_llm("★★这是……如此——好吧￥"), "这是……如此，好吧。")

    def test_ellipsis_is_kept(self):
        self.assertEqual(clean_for_llm("他走了……"), "他走了……")
        self.assertEqual(normalize_tts("等等……好啦"), "等等……好啦。")
        self.assertEqual(normalize_tts("他说...好吧"), "他说……好吧。")


class NormalizeTtsTest(unittest.TestCase):
    def test_idempotent(self):
        text = "他说……好——真的！"
        self.assertEqual(normalize_tts(normalize_tts(text)), normalize_tts(text))

    def test_ellipsis_is_kept(self):
        self.assertEqual(normalize_tts("他说……好——真的！"), "他说……好，真的！")
        self.assertEqual(normalize_tts("你……？"), "你……？")


class OneParagraphTest(unittest.TestCase):
    def test_ellipsis_is_kept(self):
        self.assertEqual(one_paragraph("等等……好啦"), "等等……好啦")
        self.assertEqual(one_paragraph("第一……第二...第三"), "第一……第二……第三")

    def test_dashes_and_decorations_stay_commas(self):
        self.assertEqual(one_paragraph("甲——乙＊＊丙"), "甲，乙，丙")


if __name__ == "__main__":
    unittest.main()
