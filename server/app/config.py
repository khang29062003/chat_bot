from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    public_ws_url: str = "ws://127.0.0.1:8000/xiaozhi/v1/"
    # device_id -> token; key "*" là token chung (chỉ dùng khi dev)
    device_tokens: dict[str, str] = {}

    anthropic_api_key: str = ""
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "low"
    llm_max_tokens: int = 2048
    max_history_turns: int = 10
    system_prompt: str = (
        "Bạn là trợ lý giọng nói thân thiện. Trả lời bằng tiếng Việt, ngắn gọn, "
        "không dùng markdown hay emoji."
    )

    stt_provider: str = "deepgram"
    deepgram_api_key: str = ""
    deepgram_model: str = "nova-2"
    stt_language: str = "vi"

    tts_provider: str = "viettel"
    viettel_tts_token: str = ""
    viettel_tts_voice: str = "hn-quynhanh"
    viettel_tts_url: str = "https://viettelai.vn/tts/speech_synthesis"
    edge_tts_voice: str = "vi-VN-HoaiMyNeural"

    # Audio
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    frame_duration_ms: int = 60
    vad_silence_ms: int = 700  # im lặng bao lâu thì coi là hết câu (mode auto)


@lru_cache
def get_settings() -> Settings:
    return Settings()
