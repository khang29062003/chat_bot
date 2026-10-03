"""Claude LLM: stream câu trả lời và cắt thành từng câu để TTS sớm."""
import re
from collections.abc import AsyncIterator

import anthropic

from .config import Settings

_SENTENCE_END = re.compile(r"(.+?[.!?。…\n]+)(\s|$)", re.S)


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Tách các câu hoàn chỉnh khỏi buffer, trả về (câu, phần còn lại)."""
    sentences = []
    while m := _SENTENCE_END.match(buffer):
        s = m.group(1).strip()
        if s:
            sentences.append(s)
        buffer = buffer[m.end() :]
    return sentences, buffer


class ClaudeLLM:
    def __init__(self, s: Settings):
        self.s = s
        self.client = anthropic.AsyncAnthropic(api_key=s.anthropic_api_key or None)

    async def stream_sentences(self, messages: list[dict]) -> AsyncIterator[str]:
        """Yield từng câu trả lời. `messages` được append câu trả lời đầy đủ khi xong."""
        buffer = ""
        async with self.client.beta.messages.stream(
            model=self.s.llm_model,
            max_tokens=self.s.llm_max_tokens,
            system=self.s.system_prompt,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.s.llm_effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            async for text in stream.text_stream:
                buffer += text
                sentences, buffer = split_sentences(buffer)
                for s in sentences:
                    yield s
            final = await stream.get_final_message()

        if buffer.strip():
            yield buffer.strip()
        if final.stop_reason == "refusal":
            yield "Xin lỗi, mình không thể trả lời câu hỏi này."
            messages.pop()  # bỏ lượt user bị từ chối để không làm hỏng lịch sử
            return
        # Append nguyên content (giữ thinking block) để hội thoại hợp lệ ở lượt sau
        messages.append({"role": "assistant", "content": final.content})
