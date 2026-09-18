"""Offline checks for the workflow overlay store (no network, temp dirs only)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import workflow_store


class WorkflowStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_defaults_include_pipeline_and_prompts(self):
        state = workflow_store.merged("nobook", self.base)
        self.assertEqual(state["pipeline"]["product"]["name"], "文本（段落流）")
        self.assertIn("local_system", state["prompts"])
        self.assertEqual(state["params"]["batch"], 5)

    def test_save_merges_partial_and_logs(self):
        workflow_store.save("b", {"params": {"batch": 9}, "prompts": {"step_mark": "自定义"}}, actor="web", base=self.base)
        state = workflow_store.merged("b", self.base)
        self.assertEqual(state["params"]["batch"], 9)
        self.assertEqual(state["prompts"]["step_mark"], "自定义")
        self.assertEqual(state["overlay"]["updated_by"], "web")
        self.assertEqual(len(state["changelog"]), 1)
        self.assertEqual(state["changelog"][0]["actor"], "web")

    def test_clamps_values_and_rejects_unknown_prompt(self):
        state = workflow_store.save("b", {"params": {"batch": 9999}}, base=self.base)
        self.assertEqual(state["params"]["batch"], 200)
        with self.assertRaises(ValueError):
            workflow_store.save("b", {"prompts": {"nope": "x"}}, base=self.base)

    def test_clear_removes_overlay(self):
        workflow_store.save("b", {"params": {"think": False}}, base=self.base)
        self.assertFalse(workflow_store.merged("b", self.base)["params"]["think"])
        state = workflow_store.clear("b", actor="agent", base=self.base)
        self.assertEqual(state["params"]["think"], state["defaults"]["params"]["think"])
        self.assertTrue(state["changelog"][0]["cleared"])


if __name__ == "__main__":
    unittest.main()
