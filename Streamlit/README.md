# F5-TTS Vietnamese Demo

Giao diện Streamlit + FastAPI cho phép tổng hợp giọng nói tiếng Việt với voice cloning từ các checkpoint F5-TTS.

## Cấu trúc project

```
Streamlit/
├── app.py                   # Streamlit UI (frontend)
├── start.sh                 # Script khởi động cả 2 service
├── requirements.txt         # Dependencies bổ sung
├── outputs/                 # File WAV kết quả sinh ra
└── backend/
    ├── __init__.py
    ├── api.py               # FastAPI endpoints
    ├── queue_manager.py     # Async job queue (ThreadPoolExecutor)
    ├── model_manager.py     # Model cache & lifecycle
    └── schemas.py           # Pydantic request/response schemas
```

## Yêu cầu

- Python 3.10+
- Môi trường đã cài đặt dependencies từ `requirements.txt` gốc của project
- CUDA GPU (khuyến nghị)

## Cài đặt dependencies bổ sung

```bash
cd /home/reg/TTS_DATA/Streamlit
pip install -r requirements.txt
```

## Khởi động

### Cách 1 — Script tự động (chạy cả 2 service)

```bash
cd /home/reg/TTS_DATA/Streamlit
bash start.sh
```

### Cách 2 — Chạy riêng từng service

**Terminal 1 — FastAPI Backend:**

```bash
cd /home/reg/TTS_DATA/Streamlit
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Streamlit Frontend:**

```bash
cd /home/reg/TTS_DATA/Streamlit
streamlit run app.py --server.port 8501
```

## Truy cập

| Service     | URL                            |
|-------------|-------------------------------|
| Frontend    | http://localhost:8501          |
| API Docs    | http://localhost:8000/docs     |
| ReDoc       | http://localhost:8000/redoc    |
| Health      | http://localhost:8000/health   |

## API Endpoints

| Method | Path                   | Mô tả                              |
|--------|------------------------|-------------------------------------|
| GET    | `/api/models`          | Danh sách checkpoint models         |
| GET    | `/api/samples`         | Danh sách reference samples         |
| POST   | `/api/tts`             | Submit TTS job → trả về `job_id`   |
| GET    | `/api/tts/{job_id}`    | Polling trạng thái / kết quả job   |
| GET    | `/outputs/{filename}`  | Tải file WAV kết quả               |

## Luồng hoạt động

```
POST /api/tts (ref_audio_b64, ref_text, target_text, model_name)
      ↓
Tạo job_id (UUID)
      ↓
Đưa vào queue.Queue
      ↓
Worker thread:
  - Decode base64 audio → temp WAV
  - Load model (cache nếu cùng checkpoint)
  - model.synthesize(target_text, ref_audio, ref_text)
  - Lưu output WAV vào outputs/
  - Cập nhật JobResult
      ↓
Client polling GET /api/tts/{job_id} mỗi 2 giây
      ↓
{ "status": "completed", "audio_url": "/outputs/xxx.wav", "duration": 12.4 }
```

## Notes

- Model được cache trong RAM/VRAM. Chỉ reload khi chọn checkpoint khác.
- Khi đổi model: unload model cũ → giải phóng VRAM → load model mới.
- Worker pool chỉ có 1 worker để tránh OOM khi chạy nhiều job song song.
- File WAV tạm thời (ref audio) được xóa sau khi synthesize xong.
