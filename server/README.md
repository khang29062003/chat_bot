# Chat bot server

Server giọng nói tương thích firmware [xiaozhi-esp32](https://github.com/78/xiaozhi-esp32) (WebSocket protocol v1).

```
ESP32-S3 ──Opus 16kHz──▶ VAD ─▶ STT (Deepgram) ─▶ Claude (stream) ─▶ tách câu ─▶ TTS (Viettel AI) ─▶ Opus 24kHz ──▶ loa
```

## Chạy

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
cp .env.example .env                            # rồi điền API key
.venv/Scripts/uvicorn app.main:app --host 0.0.0.0 --port 8000
.venv/Scripts/python -m pytest                  # test với provider giả
```

Docker: `docker build -t chatbot-server . && docker run --env-file .env -p 8000:8000 chatbot-server`

## Cấu hình firmware

Trong `menuconfig` của xiaozhi-esp32, đặt **OTA URL** là `http://<server>:8000/xiaozhi/ota/`.
Thiết bị gọi OTA lúc khởi động và nhận về `PUBLIC_WS_URL` cùng token.

## Endpoint

| Endpoint | Mô tả |
|---|---|
| `POST /xiaozhi/ota/` | Trả về địa chỉ WebSocket, token và thông tin firmware |
| `WS /xiaozhi/v1/` | Hội thoại. Header `Authorization: Bearer <token>`, `Device-Id: <MAC>` |
| `GET /health` | Health check |

## Cấu trúc

- `app/session.py`: máy trạng thái của phiên (hello / listen / abort), pipeline STT → LLM → TTS
- `app/llm.py`: Claude streaming, cắt câu để TTS sớm, bật refusal fallback
- `app/stt.py`, `app/tts.py`: provider có thể thay thế (thêm provider mới bằng cách implement `transcribe` / `synthesize`)
- `app/audio.py`: Opus codec (PyAV), VAD (webrtcvad)

## Ghi chú

- `EdgeTTS` chỉ dùng khi dev. Dịch vụ không chính thức nên không dùng cho sản phẩm thương mại.
- Endpoint và tham số của Viettel TTS cần đối chiếu lại với tài liệu trong hợp đồng.
- Lịch sử hội thoại hiện lưu trong RAM theo từng phiên.
