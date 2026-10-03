"""Text-to-speech providers. Mỗi provider trả về PCM s16 mono ở sample_rate yêu cầu."""
from typing import Protocol

import httpx

from .audio import decode_audio_to_pcm
from .config import Settings


class TTS(Protocol):
    async def synthesize(self, text: str, sample_rate: int) -> bytes: ...


class ViettelTTS:
    """Viettel AI TTS. Kiểm tra lại endpoint/tham số theo tài liệu hợp đồng của bạn."""

    def __init__(self, s: Settings):
        self.s = s
        self.http = httpx.AsyncClient(timeout=30)

    async def synthesize(self, text: str, sample_rate: int) -> bytes:
        r = await self.http.post(
            self.s.viettel_tts_url,
            json={
                "text": text,
                "voice": self.s.viettel_tts_voice,
                "speed": 1.0,
                "tts_return_option": 2,  # 2 = wav, 3 = mp3
                "token": self.s.viettel_tts_token,
                "without_filter": False,
            },
        )
        r.raise_for_status()
        return decode_audio_to_pcm(r.content, sample_rate)


class EdgeTTS:
    """Microsoft Edge TTS (miễn phí, không chính thức) — CHỈ dùng khi phát triển."""

    def __init__(self, s: Settings):
        self.voice = s.edge_tts_voice

    async def synthesize(self, text: str, sample_rate: int) -> bytes:
        import edge_tts

        mp3 = bytearray()
        async for chunk in edge_tts.Communicate(text, self.voice).stream():
            if chunk["type"] == "audio":
                mp3 += chunk["data"]
        return decode_audio_to_pcm(bytes(mp3), sample_rate)


def create_tts(s: Settings) -> TTS:
    if s.tts_provider == "viettel":
        return ViettelTTS(s)
    if s.tts_provider == "edge":
        return EdgeTTS(s)
    raise ValueError(f"Unknown TTS provider: {s.tts_provider}")
