"""Test end-to-end với STT/TTS/LLM giả (không gọi API thật)."""
import json

import numpy as np
from fastapi.testclient import TestClient

from app import main
from app.audio import OpusDecoder, OpusEncoder
from app.llm import split_sentences


def test_split_sentences():
    s, rest = split_sentences("Xin chào! Mình là bot. Còn dở")
    assert s == ["Xin chào!", "Mình là bot."]
    assert rest == "Còn dở"


def test_opus_roundtrip():
    tone = (np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000) * 8000).astype(np.int16).tobytes()
    packets = OpusEncoder(16000, 60).encode(tone)
    assert len(packets) >= 16
    dec = OpusDecoder(16000)
    pcm = b"".join(dec.decode(p) for p in packets)
    assert len(pcm) > 16000  # ~1s audio


class FakeSTT:
    async def transcribe(self, pcm, sr):
        return "xin chào"


class FakeTTS:
    async def synthesize(self, text, sr):
        return b"\x00\x00" * (sr // 5)  # 200ms im lặng


class FakeLLM:
    async def stream_sentences(self, messages):
        yield "Chào bạn!"
        yield "Mình giúp gì được?"
        messages.append({"role": "assistant", "content": "Chào bạn! Mình giúp gì được?"})


def test_conversation(monkeypatch):
    monkeypatch.setattr(main, "stt", FakeSTT())
    monkeypatch.setattr(main, "tts", FakeTTS())
    monkeypatch.setattr(main, "llm", FakeLLM())
    monkeypatch.setattr(main.settings, "device_tokens", {"*": "t"})

    speech = (np.random.default_rng(0).normal(0, 3000, 16000)).astype(np.int16).tobytes()
    packets = OpusEncoder(16000, 60).encode(speech)

    client = TestClient(main.app)
    with client.websocket_connect("/xiaozhi/v1/", headers={"Authorization": "Bearer t", "Device-Id": "aa:bb"}) as ws:
        ws.send_text(json.dumps({"type": "hello", "version": 1, "transport": "websocket",
                                 "audio_params": {"format": "opus", "sample_rate": 16000, "channels": 1, "frame_duration": 60}}))
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "hello" and hello["audio_params"]["sample_rate"] == 24000

        ws.send_text(json.dumps({"type": "listen", "state": "start", "mode": "manual"}))
        for p in packets:
            ws.send_bytes(p)
        ws.send_text(json.dumps({"type": "listen", "state": "stop"}))

        events, audio_frames = [], 0
        while True:
            m = ws.receive()
            if m.get("bytes"):
                audio_frames += 1
                continue
            msg = json.loads(m["text"])
            events.append((msg["type"], msg.get("state"), msg.get("text")))
            if msg["type"] == "tts" and msg["state"] == "stop":
                break

    assert ("stt", None, "xin chào") in events
    assert ("tts", "sentence_start", "Chào bạn!") in events
    assert audio_frames >= 6


def test_rejects_bad_token(monkeypatch):
    monkeypatch.setattr(main.settings, "device_tokens", {"*": "t"})
    client = TestClient(main.app)
    import pytest
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/xiaozhi/v1/", headers={"Authorization": "Bearer wrong"}) as ws:
            ws.receive_text()
