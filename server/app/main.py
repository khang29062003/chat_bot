import logging
import secrets

from fastapi import FastAPI, Request, WebSocket, status

from .config import get_settings
from .llm import ClaudeLLM
from .session import DeviceSession
from .stt import create_stt
from .tts import create_tts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("chatbot")

settings = get_settings()
app = FastAPI(title="OhStem Chat Bot Server")
stt = create_stt(settings)
tts = create_tts(settings)
llm = ClaudeLLM(settings)


def check_token(device_id: str, auth_header: str | None) -> bool:
    if not settings.device_tokens:
        return True  # chưa cấu hình token: chế độ dev
    expected = settings.device_tokens.get(device_id) or settings.device_tokens.get("*")
    token = (auth_header or "").removeprefix("Bearer ").strip()
    return bool(expected) and secrets.compare_digest(token, expected)


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/xiaozhi/ota/")
async def ota(request: Request):
    """Firmware xiaozhi gọi endpoint này khi khởi động để lấy địa chỉ WebSocket."""
    device_id = request.headers.get("Device-Id", "")
    token = settings.device_tokens.get(device_id) or settings.device_tokens.get("*", "")
    log.info("OTA check from %s", device_id)
    return {
        "websocket": {"url": settings.public_ws_url, "token": token},
        "firmware": {"version": "0.0.0", "url": ""},  # TODO: phát hành firmware OTA
    }


@app.websocket("/xiaozhi/v1/")
async def ws_endpoint(ws: WebSocket):
    device_id = ws.headers.get("device-id") or ws.query_params.get("device-id", "unknown")
    if not check_token(device_id, ws.headers.get("authorization")):
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await ws.accept()
    log.info("device connected: %s", device_id)
    await DeviceSession(ws, device_id, settings, stt, tts, llm).run()
    log.info("device disconnected: %s", device_id)
