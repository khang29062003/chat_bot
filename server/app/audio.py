"""Opus encode/decode, resample và VAD."""
from fractions import Fraction

import av
import numpy as np
import webrtcvad


class OpusDecoder:
    """Giải mã các gói Opus thô (mỗi WebSocket binary frame = 1 gói) sang PCM s16 mono."""

    def __init__(self, sample_rate: int):
        self.ctx = av.CodecContext.create("libopus", "r")
        self.ctx.sample_rate = sample_rate
        self.ctx.layout = "mono"
        self.resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)

    def decode(self, packet: bytes) -> bytes:
        out = bytearray()
        for frame in self.ctx.decode(av.Packet(packet)):
            for f in self.resampler.resample(frame):
                out += f.to_ndarray().tobytes()
        return bytes(out)


class OpusEncoder:
    """Mã hóa PCM s16 mono thành các gói Opus có độ dài frame cố định."""

    def __init__(self, sample_rate: int, frame_ms: int):
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000
        self.ctx = av.CodecContext.create("libopus", "w")
        self.ctx.sample_rate = sample_rate
        self.ctx.layout = "mono"
        self.ctx.format = "s16"
        self.ctx.bit_rate = 32000
        self.ctx.time_base = Fraction(1, sample_rate)
        self.ctx.options = {"frame_duration": str(frame_ms), "application": "voip"}
        self.ctx.open()
        self._pts = 0

    def encode(self, pcm: bytes) -> list[bytes]:
        """pcm phải có độ dài bội số của frame; phần thiếu được đệm 0."""
        samples = np.frombuffer(pcm, dtype=np.int16)
        rem = len(samples) % self.frame_samples
        if rem:
            samples = np.concatenate([samples, np.zeros(self.frame_samples - rem, np.int16)])
        packets: list[bytes] = []
        for i in range(0, len(samples), self.frame_samples):
            chunk = samples[i : i + self.frame_samples].reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(chunk, format="s16", layout="mono")
            frame.sample_rate = self.sample_rate
            frame.pts = self._pts
            self._pts += self.frame_samples
            packets += [bytes(p) for p in self.ctx.encode(frame)]
        return packets


def decode_audio_to_pcm(data: bytes, sample_rate: int) -> bytes:
    """Giải mã file audio bất kỳ (mp3/wav/...) sang PCM s16 mono ở sample_rate."""
    import io

    out = bytearray()
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    with av.open(io.BytesIO(data)) as container:
        for frame in container.decode(audio=0):
            for f in resampler.resample(frame):
                out += f.to_ndarray().tobytes()
    for f in resampler.resample(None):
        out += f.to_ndarray().tobytes()
    return bytes(out)


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class EndOfSpeechDetector:
    """VAD đơn giản: báo hết câu khi đã có tiếng nói rồi im lặng đủ lâu."""

    FRAME_MS = 30

    def __init__(self, sample_rate: int, silence_ms: int, aggressiveness: int = 2):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_bytes = sample_rate * self.FRAME_MS // 1000 * 2
        self.silence_frames_needed = silence_ms // self.FRAME_MS
        self.reset()

    def reset(self) -> None:
        self._buf = b""
        self.heard_speech = False
        self._silence = 0

    def feed(self, pcm: bytes) -> bool:
        self._buf += pcm
        ended = False
        while len(self._buf) >= self.frame_bytes:
            frame, self._buf = self._buf[: self.frame_bytes], self._buf[self.frame_bytes :]
            if self.vad.is_speech(frame, self.sample_rate):
                self.heard_speech = True
                self._silence = 0
            elif self.heard_speech:
                self._silence += 1
                if self._silence >= self.silence_frames_needed:
                    ended = True
        return ended
