# TTS_DATA

Dự án thu thập, tiền xử lý và đánh giá dữ liệu cho hệ thống Text-to-Speech tiếng Việt.

## Cấu trúc thư mục

```
TTS-DATA/
├── Craw_data/          # Scripts thu thập dữ liệu (YouTube, ...)
├── Preprocess/         # Scripts tiền xử lý âm thanh & văn bản
├── Processed_DATA/     # Dữ liệu đã xử lý (không track trong git)
├── F5-TTS-Vietnamese/  # Submodule / code F5-TTS
├── components/         # Các module dùng chung
├── evaluate/           # Scripts đánh giá model (WER, CER, MCD, RTF)
├── models/             # Model weights (không track trong git)
├── requirements.txt          # Dependencies đầy đủ
└── requirements_crawler.txt  # Dependencies cho crawler
```

## Cài đặt

```bash
pip install -r requirements.txt
# Hoặc chỉ crawler:
pip install -r requirements_crawler.txt
```

## Pipeline

1. **Thu thập dữ liệu** — `Craw_data/` — crawl YouTube, audiobook, ...
2. **Tiền xử lý** — `Preprocess/` — ASR, normalize, filter audio
3. **Đánh giá** — `evaluate/` — đo WER/CER, MCD, RTF với F5-TTS
