"""Một phiên WebSocket với thiết bị theo giao thức xiaozhi-esp32 (protocol v1).

Luồng:
  device -> hello                     server -> hello (session_id, audio_params)
  device -> listen start              (bắt đầu thu âm, gửi các gói Opus nhị phân)
  device -> listen stop  | VAD auto   server -> stt {text}
                                      server -> llm {emotion}
                                      server -> tts start / sentence_start / [opus...] / stop
  device -> abort                     (ngắt lời: dừng TTS ngay)
"""
import asyncio
import json
import logging
import time
import uuid

from fastapi import WebSocket, WebSocketDisconnect

from .audio import EndOfSpeechDetector, OpusDecoder, OpusEncoder
from .config import Settings
from .llm import ClaudeLLM
from .stt import STT
from .tts import TTS

log = logging.getLogger(__name__)


class DeviceSession:
    def __init__(self, ws: WebSocket, device_id: str, s: Settings, stt: STT, tts: TTS, llm: ClaudeLLM):
        self.ws, self.device_id, self.s = ws, device_id, s
        self.stt, self.tts, self.llm = stt, tts, llm
        self.session_id = str(uuid.uuid4())
        self.in_rate = s.input_sample_rate
        self.decoder = OpusDecoder(self.in_rate)
        self.vad = EndOfSpeechDetector(self.in_rate, s.vad_silence_ms)
        self.listening = False
        self.listen_mode = "auto"
        self.pcm = bytearray()
        self.history: list[dict] = []
        self.reply_task: asyncio.Task | None = None

    async def send_json(self, msg: dict) -> None:
        await self.ws.send_text(json.dumps({"session_id": self.session_id, **msg}, ensure_ascii=False))

    async def run(self) -> None:
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await self.on_audio(msg["bytes"])
                elif msg.get("text") is not None:
                    await self.on_text(json.loads(msg["text"]))
        except WebSocketDisconnect:
            pass
        finally:
            await self.cancel_reply()

    # ---------- control messages ----------
    async def on_text(self, msg: dict) -> None:
        t = msg.get("type")
        if t == "hello":
            params = msg.get("audio_params") or {}
            if params.get("sample_rate") and params["sample_rate"] != self.in_rate:
                self.in_rate = params["sample_rate"]
                self.decoder = OpusDecoder(self.in_rate)
                self.vad = EndOfSpeechDetector(self.in_rate, self.s.vad_silence_ms)
            await self.send_json({
                "type": "hello",
                "transport": "websocket",
                "audio_params": {
                    "format": "opus",
                    "sample_rate": self.s.output_sample_rate,
                    "channels": 1,
                    "frame_duration": self.s.frame_duration_ms,
                },
            })
        elif t == "listen":
            state = msg.get("state")
            if state == "start":
                await self.cancel_reply()
                self.listen_mode = msg.get("mode", "auto")
                self.start_listening()
            elif state == "stop":
                await self.finish_utterance()
            elif state == "detect":  # wake word được phát hiện trên thiết bị
                log.info("[%s] wake word: %s", self.device_id, msg.get("text"))
        elif t == "abort":
            await self.cancel_reply()
            await self.send_json({"type": "tts", "state": "stop"})
        else:
            log.debug("[%s] ignore message %s", self.device_id, t)

    def start_listening(self) -> None:
        self.listening = True
        self.pcm.clear()
        self.vad.reset()

    # ---------- audio ----------
    async def on_audio(self, packet: bytes) -> None:
        if not self.listening:
            return
        try:
            pcm = self.decoder.decode(packet)
        except Exception:
            log.warning("[%s] bad opus packet", self.device_id)
            return
        self.pcm += pcm
        if self.listen_mode == "auto" and self.vad.feed(pcm):
            await self.finish_utterance()

    async def finish_utterance(self) -> None:
        if not self.listening:
            return
        self.listening = False
        pcm, self.pcm = bytes(self.pcm), bytearray()
        if len(pcm) < self.in_rate * 2 * 0.3:  # < 300ms: bỏ qua
            return
        self.reply_task = asyncio.create_task(self.respond(pcm))

    async def cancel_reply(self) -> None:
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()
            try:
                await self.reply_task
            except asyncio.CancelledError:
                pass
        self.reply_task = None

    # ---------- pipeline STT -> LLM -> TTS ----------
    async def respond(self, pcm: bytes) -> None:
        try:
            text = await self.stt.transcribe(pcm, self.in_rate)
            log.info("[%s] user: %s", self.device_id, text)
            await self.send_json({"type": "stt", "text": text})
            if not text:
                if self.listen_mode == "auto":
                    self.start_listening()
                return

            await self.send_json({"type": "llm", "text": "🙂", "emotion": "happy"})
            await self.send_json({"type": "tts", "state": "start"})
            self.history.append({"role": "user", "content": text})
            self.trim_history()

            encoder = OpusEncoder(self.s.output_sample_rate, self.s.frame_duration_ms)
            async for sentence in self.llm.stream_sentences(self.history):
                log.info("[%s] bot: %s", self.device_id, sentence)
                await self.send_json({"type": "tts", "state": "sentence_start", "text": sentence})
                audio = await self.tts.synthesize(sentence, self.s.output_sample_rate)
                await self.stream_opus(encoder.encode(audio))
                await self.send_json({"type": "tts", "state": "sentence_end", "text": sentence})
            await self.send_json({"type": "tts", "state": "stop"})
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[%s] pipeline error", self.device_id)
            await self.send_json({"type": "tts", "state": "stop"})

    async def stream_opus(self, packets: list[bytes]) -> None:
        """Gửi theo nhịp thời gian thực (đi trước ~3 frame) để thiết bị không tràn buffer."""
        frame_s = self.s.frame_duration_ms / 1000
        start = time.monotonic()
        for i, p in enumerate(packets):
            ahead = start + (i - 3) * frame_s - time.monotonic()
            if ahead > 0:
                await asyncio.sleep(ahead)
            await self.ws.send_bytes(p)

    def trim_history(self) -> None:
        """Giữ tối đa N lượt; luôn bắt đầu bằng một lượt user dạng text."""
        max_msgs = self.s.max_history_turns * 2
        if len(self.history) > max_msgs:
            del self.history[: len(self.history) - max_msgs]
            while self.history and self.history[0]["role"] != "user":
                self.history.pop(0)
