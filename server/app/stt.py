"""Speech-to-text providers."""
from typing import Protocol

import httpx

from .audio import pcm_to_wav
from .config import Settings


class STT(Protocol):
    async def transcribe(self, pcm: bytes, sample_rate: int) -> str: ...


class DeepgramSTT:
    URL = "https://api.deepgram.com/v1/listen"

    def __init__(self, s: Settings):
        self.s = s
        self.http = httpx.AsyncClient(timeout=30)

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        r = await self.http.post(
            self.URL,
            params={"model": self.s.deepgram_model, "language": self.s.stt_language, "smart_format": "true"},
            headers={"Authorization": f"Token {self.s.deepgram_api_key}", "Content-Type": "audio/wav"},
            content=pcm_to_wav(pcm, sample_rate),
        )
        r.raise_for_status()
        alts = r.json()["results"]["channels"][0]["alternatives"]
        return alts[0]["transcript"].strip() if alts else ""


def create_stt(s: Settings) -> STT:
    if s.stt_provider == "deepgram":
        return DeepgramSTT(s)
    raise ValueError(f"Unknown STT provider: {s.stt_provider}")
