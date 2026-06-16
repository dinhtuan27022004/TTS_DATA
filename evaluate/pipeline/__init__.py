"""
Pipeline module cho hệ thống đánh giá F5-TTS.

3 giai đoạn:
  Phase 1 – Synthesis:  N process song song, mỗi process = 1 checkpoint
  Phase 2 – Metrics:    Tính PESQ/STOI/UTMOS/F0/WER/CER sau khi synthesis xong
  Phase 3 – Charts:     Vẽ đồ thị so sánh bằng seaborn
"""
