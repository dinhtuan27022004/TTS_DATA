#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — Khởi động F5-TTS Demo (FastAPI + Streamlit) trong 2 terminal
#
# Cách dùng:
#   cd /home/reg/TTS_DATA/Streamlit
#   bash start.sh
#
# Hoặc chạy riêng từng service:
#   Terminal 1: uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
#   Terminal 2: streamlit run app.py --server.port 8501
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Tạo outputs dir nếu chưa có
mkdir -p outputs

echo "╔══════════════════════════════════════════════════╗"
echo "║         F5-TTS Vietnamese Demo Launcher          ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "📡 Khởi động FastAPI backend (port 8000)..."
uvicorn backend.api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "   → PID: $BACKEND_PID"

sleep 2

echo ""
echo "🌐 Khởi động Streamlit frontend (port 8501)..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!
echo "   → PID: $FRONTEND_PID"

echo ""
echo "✅ Cả hai service đã khởi động!"
echo "   API:       http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo "   Frontend:  http://localhost:8501"
echo ""
echo "Nhấn Ctrl+C để dừng cả hai service."
echo ""

# Cleanup on exit
trap "echo 'Stopping...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

wait $BACKEND_PID $FRONTEND_PID
