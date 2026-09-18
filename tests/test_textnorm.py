"""Pure-function checks for the pre-LLM TTS character filter (no network)."""

from __future__ import annotations

import unittest

from audiobook.textnorm import filter_tts_chars, keep_layout


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


class KeepLayoutTest(unittest.TestCase):
    def test_preserves_paragraphs_indentation_and_punctuation(self):
        source = "第一段——有引号“是的”和省略……※。\n  第二段！\n\n[笑]<测试>"
        self.assertEqual(keep_layout(source), "第一段——有引号“是的”和省略……※。\n  第二段！\n\n笑测试")

    def test_folds_ascii_ellipsis_only(self):
        self.assertEqual(keep_layout("他说...好吧"), "他说……好吧")
        self.assertEqual(keep_layout("甲——乙"), "甲——乙")

    def test_drops_only_our_delimiters(self):
        self.assertEqual(keep_layout("保留【方头括号】｛花括号｝《书名号》"), "保留【方头括号】｛花括号｝《书名号》")


if __name__ == "__main__":
    unittest.main()
