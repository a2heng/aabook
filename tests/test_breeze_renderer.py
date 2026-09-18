"""Offline checks for the Breeze TTS 2 renderer (no network, no model)."""

from __future__ import annotations

import io
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from audiobook.tts import BreezeRenderer, _multipart, wav_bytes
from audiobook.schema import ScriptRow


def _fake_response(pcm: bytes, sample_rate: int = 24000):
    response = Mock()
    response.status = 200
    response.headers = {"X-Sample-Rate": str(sample_rate)}
    response.read.return_value = pcm
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    return response


class WavBytesTest(unittest.TestCase):
    def test_float_wav_becomes_mono_pcm16(self):
        path = "/tmp/opencode/_breeze_ref_float.wav"
        tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(2400) / 24000)
        stereo = np.stack([tone, tone], axis=1).astype(np.float32)
        sf.write(path, stereo, 24000, subtype="FLOAT")
        payload = wav_bytes(path)
        self.assertEqual(payload[:4], b"RIFF")
        data, sample_rate = sf.read(io.BytesIO(payload), dtype="int16", always_2d=True)
        self.assertEqual(sample_rate, 24000)
        self.assertEqual(data.shape[1], 1)  # mono
        self.assertEqual(data.dtype, np.dtype("int16"))


class MultipartTest(unittest.TestCase):
    def test_fields_and_file_are_framed(self):
        body, content_type = _multipart({"text": "你好"}, {"ref_audio": ("ref.wav", b"RIFF")})
        self.assertIn("multipart/form-data; boundary=", content_type)
        self.assertIn('name="text"', body.decode("utf-8"))
        self.assertIn("你好".encode(), body)
        self.assertIn('filename="ref.wav"', body.decode("utf-8"))
        self.assertIn(b"RIFF", body)
        self.assertTrue(body.endswith(b"--\r\n"))


class FailClosedTest(unittest.TestCase):
    def test_clone_without_transcript_raises_before_upload(self):
        renderer = BreezeRenderer(log=lambda _message: None)
        row = ScriptRow(seg_id="t", tts_text="你好", role_name="张三")
        with patch.object(renderer, "_post") as post:
            with self.assertRaisesRegex(RuntimeError, "transcript"):
                renderer.synth(row, "reference.wav", "")
            post.assert_not_called()

    def test_missing_voice_raises(self):
        renderer = BreezeRenderer(log=lambda _message: None)
        row = ScriptRow(seg_id="t", tts_text="你好", role_name="张三")
        with self.assertRaisesRegex(RuntimeError, "reference voice"):
            renderer.synth(row, "", "")

    def test_design_requires_instruction(self):
        renderer = BreezeRenderer(log=lambda _message: None)
        with self.assertRaisesRegex(ValueError, "instruction"):
            renderer.design_voice("你好", "  ")

    def test_asr_failure_is_reported(self):
        renderer = BreezeRenderer(log=lambda _message: None)
        renderer._asr = Mock()
        renderer._asr.transcribe.side_effect = RuntimeError("model missing")
        with self.assertRaisesRegex(RuntimeError, "ASR failed"):
            renderer.ref_text_for("reference.wav", "张三")


class TranscriptTest(unittest.TestCase):
    def test_metadata_transcript_wins(self):
        renderer = BreezeRenderer(voice_meta={"张三": {"ref_text": "参考文字"}}, log=lambda _message: None)
        self.assertEqual(renderer.ref_text_for("reference.wav", "张三"), "参考文字")

    def test_metadata_must_match_role_not_only_path(self):
        renderer = BreezeRenderer(voice_meta={"李四": {"ref_text": "别人的"}}, log=lambda _message: None)
        renderer._ref_texts["reference.wav"] = "本声的"
        self.assertEqual(renderer.ref_text_for("reference.wav", "张三"), "本声的")


class PostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = "/tmp/opencode/_breeze_ref.wav"
        tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(2400) / 24000)
        sf.write(cls.ref, tone.astype(np.float32), 24000, subtype="FLOAT")

    def test_clone_uploads_ascii_name_without_instruction(self):
        renderer = BreezeRenderer(log=lambda _message: None)
        row = ScriptRow(seg_id="t", tts_text="你好", role_name="张三")
        payload = np.zeros(240, dtype="<i2").tobytes()
        captured = {}

        def fake_open(request, timeout=None):
            captured["body"] = request.data
            captured["url"] = request.full_url
            return _fake_response(payload)

        with patch("audiobook.tts._OPENER.open", side_effect=fake_open):
            samples, sample_rate = renderer.synth(row, self.ref, "参考文字")
        body = captured["body"]
        self.assertIn(b'filename="ref.wav"', body)
        self.assertNotIn(b'name="instruction"', body)
        self.assertEqual(sample_rate, 24000)
        self.assertEqual(samples.shape[0], 240)

    def test_direction_sends_instruction(self):
        renderer = BreezeRenderer(log=lambda _message: None)
        renderer.config.direction = True
        row = ScriptRow(seg_id="t", tts_text="你好", role_name="张三", style_desc="低沉克制")
        payload = np.zeros(240, dtype="<i2").tobytes()
        captured = {}
        with patch(
            "audiobook.tts._OPENER.open",
            side_effect=lambda request, timeout=None: (captured.update(body=request.data) or _fake_response(payload)),
        ):
            renderer.synth(row, self.ref, "参考文字")
        self.assertIn(b'name="instruction"', captured["body"])

    def test_odd_pcm_is_rejected(self):
        renderer = BreezeRenderer(log=lambda _message: None)
        row = ScriptRow(seg_id="t", tts_text="你好", role_name="张三")
        with patch("audiobook.tts._OPENER.open", return_value=_fake_response(b"\x01")):
            with self.assertRaisesRegex(RuntimeError, "invalid PCM"):
                renderer.synth(row, self.ref, "参考文字")


if __name__ == "__main__":
    unittest.main()
