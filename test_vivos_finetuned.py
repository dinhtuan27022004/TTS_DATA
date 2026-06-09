"""
Script test model F5-TTS mới finetune trên dataset VIVOS.

Cách chạy:
    # Test model finetuned (default)
    python test_vivos_finetuned.py

    # So sánh finetuned vs V0
    python test_vivos_finetuned.py --model both

    # Chỉ test V0 (baseline)
    python test_vivos_finetuned.py --model v0

    # Giới hạn số mẫu để chạy nhanh
    python test_vivos_finetuned.py --num-samples 50

    # Chỉ dùng test split
    python test_vivos_finetuned.py --split test

    # Chỉ định đường dẫn VIVOS thủ công
    python test_vivos_finetuned.py --vivos-dir /path/to/vivos

Dataset VIVOS format được hỗ trợ:
    vivos/
        train/
            waves/<speaker>/<speaker>_<id>.wav
            prompts.txt  (format: "SPEAKER_ID text content")
        test/
            waves/<speaker>/<speaker>_<id>.wav
            prompts.txt

Kết quả lưu tại: evaluate/results/  và  outputs/test_vivos/
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_vivos")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────
# Cấu hình mặc định
# ─────────────────────────────────────────────
DEFAULT_VIVOS_DIRS = [
    os.path.join(PROJECT_ROOT, "data", "archive", "vivos"),
    os.path.join(PROJECT_ROOT, "data", "vivos"),
    os.path.join(PROJECT_ROOT, "Craw_data", "vivos"),
    os.path.join(os.path.expanduser("~"), "Downloads", "vivos"),
    os.path.join(os.path.expanduser("~"), "vivos"),
    "/tmp/vivos",
]
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "test_vivos")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "evaluate", "results")

# ─────────────────────────────────────────────
# VIVOS Dataset Loader
# ─────────────────────────────────────────────

@dataclass
class VIVOSSample:
    sample_id: str
    speaker: str
    text: str
    wav_path: str
    split: str


def find_vivos_dir(custom_path: Optional[str] = None) -> Optional[str]:
    """Tự động tìm thư mục VIVOS."""
    candidates = ([custom_path] if custom_path else []) + DEFAULT_VIVOS_DIRS
    for d in candidates:
        if d and os.path.isdir(d):
            # Kiểm tra có chứa train/ hoặc test/
            if any(os.path.isdir(os.path.join(d, s)) for s in ("train", "test")):
                return d
    return None


def load_vivos_split(vivos_dir: str, split: str) -> List[VIVOSSample]:
    """
    Parse một split (train/test) của VIVOS.

    Format prompts.txt:
        VIVOSSPK01_R001 nội dung text câu nói
    """
    split_dir = os.path.join(vivos_dir, split)
    prompts_file = os.path.join(split_dir, "prompts.txt")
    waves_dir = os.path.join(split_dir, "waves")

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Không tìm thấy split '{split}' tại: {split_dir}")
    if not os.path.isfile(prompts_file):
        raise FileNotFoundError(f"Không tìm thấy prompts.txt: {prompts_file}")

    samples = []
    missing = 0

    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue

            sample_id, text = parts[0].strip(), parts[1].strip()
            if not sample_id or not text:
                continue

            # speaker = phần đầu của sample_id (VIVOSSPK01)
            speaker = "_".join(sample_id.split("_")[:-1]) if "_" in sample_id else sample_id

            wav_path = os.path.join(waves_dir, speaker, f"{sample_id}.wav")
            if not os.path.isfile(wav_path):
                missing += 1
                continue

            samples.append(VIVOSSample(
                sample_id=sample_id,
                speaker=speaker,
                text=text,
                wav_path=wav_path,
                split=split,
            ))

    logger.info(f"[VIVOS/{split}] Tải {len(samples)} mẫu (bỏ qua {missing} file thiếu)")
    return samples


def load_vivos(vivos_dir: str, splits: List[str]) -> List[VIVOSSample]:
    samples = []
    for split in splits:
        try:
            samples.extend(load_vivos_split(vivos_dir, split))
        except FileNotFoundError as e:
            logger.warning(str(e))
    return samples


# ─────────────────────────────────────────────
# Result + Persistence
# ─────────────────────────────────────────────

@dataclass
class SampleResult:
    sample_id: str
    speaker: str
    split: str
    text: str
    model_name: str
    success: bool
    error: str = ""
    mcd: Optional[float] = None
    pesq: Optional[float] = None
    stoi: Optional[float] = None
    utmos: Optional[float] = None
    f0_corr: Optional[float] = None
    wer: Optional[float] = None
    cer: Optional[float] = None
    synth_time_s: float = 0.0
    audio_duration_s: float = 0.0
    rtf: float = 0.0          # Real-Time Factor = synth_time / audio_duration


def save_results_csv(results: List[SampleResult], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not results:
        return
    fields = list(asdict(results[0]).keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    logger.info(f"CSV lưu tại: {csv_path}")


def load_resume_cache(cache_path: str) -> Dict[str, SampleResult]:
    """Load kết quả đã tính để resume."""
    if not os.path.isfile(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cache = {}
        for entry in data:
            key = f"{entry['model_name']}::{entry['sample_id']}"
            cache[key] = SampleResult(**entry)
        logger.info(f"Resume: tìm thấy {len(cache)} kết quả đã tính")
        return cache
    except Exception as e:
        logger.warning(f"Không thể đọc cache resume: {e}")
        return {}


def save_resume_cache(results: List[SampleResult], cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# Compute Metrics
# ─────────────────────────────────────────────

def compute_metrics(
    ref_wav: np.ndarray,
    syn_wav: np.ndarray,
    ref_sr: int,
    syn_sr: int,
    text: str,
    enabled_metrics: List[str],
) -> Dict[str, Optional[float]]:
    """Tính các metric. Mỗi metric được bọc try/except riêng."""
    from evaluate.metrics.calculator import MetricCalculator
    import librosa

    result = {m: None for m in ["mcd", "pesq", "stoi", "utmos", "f0_corr", "wer", "cer"]}

    # Resample cả hai về 24kHz để so sánh công bằng
    target_sr = 24000
    if ref_sr != target_sr:
        ref_wav = librosa.resample(ref_wav, orig_sr=ref_sr, target_sr=target_sr)
    if syn_sr != target_sr:
        syn_wav = librosa.resample(syn_wav, orig_sr=syn_sr, target_sr=target_sr)

    try:
        calc = MetricCalculator()
        mr = calc.compute_all(
            ref_audio=ref_wav,
            syn_audio=syn_wav,
            sr=target_sr,
            text=text,
            sample_id="tmp",
        )
        if "mcd" in enabled_metrics:
            result["mcd"] = mr.mcd
        if "pesq" in enabled_metrics:
            result["pesq"] = mr.pesq
        if "stoi" in enabled_metrics:
            result["stoi"] = mr.stoi
        if "utmos" in enabled_metrics:
            result["utmos"] = mr.utmos
        if "f0_corr" in enabled_metrics:
            result["f0_corr"] = mr.f0_correlation
        if "wer" in enabled_metrics:
            result["wer"] = mr.wer
        if "cer" in enabled_metrics:
            result["cer"] = mr.cer
    except Exception as e:
        logger.warning(f"Lỗi tính metric: {e}")

    return result


# ─────────────────────────────────────────────
# Print Summary
# ─────────────────────────────────────────────

def print_summary(results: List[SampleResult], model_name: str) -> None:
    ok = [r for r in results if r.success and r.model_name == model_name]
    if not ok:
        print(f"\n[{model_name}] Không có mẫu nào thành công.")
        return

    def mean_of(attr):
        vals = [getattr(r, attr) for r in ok if getattr(r, attr) is not None]
        return np.mean(vals) if vals else None

    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    print(f"\n{'='*60}")
    print(f"KẾT QUẢ: {model_name}  ({len(ok)}/{len(results)} mẫu thành công)")
    print(f"{'='*60}")
    print(f"  MCD          : {fmt(mean_of('mcd'))}  (↓ thấp hơn = tốt hơn)")
    print(f"  PESQ         : {fmt(mean_of('pesq'))}  (↑ cao hơn = tốt hơn, max 4.5)")
    print(f"  STOI         : {fmt(mean_of('stoi'))}  (↑ cao hơn = tốt hơn, max 1.0)")
    print(f"  UTMOS        : {fmt(mean_of('utmos'))}  (↑ cao hơn = tốt hơn, max 5.0)")
    print(f"  F0 Corr      : {fmt(mean_of('f0_corr'))}  (↑ cao hơn = tốt hơn)")
    print(f"  WER          : {fmt(mean_of('wer'))}  (↓ thấp hơn = tốt hơn)")
    print(f"  CER          : {fmt(mean_of('cer'))}  (↓ thấp hơn = tốt hơn)")
    print(f"  RTF (avg)    : {fmt(mean_of('rtf'))}  (↓ thấp hơn = nhanh hơn)")
    print(f"{'='*60}")

    # Thống kê theo speaker
    speakers = sorted(set(r.speaker for r in ok))
    if len(speakers) > 1:
        print(f"\nTheo speaker ({len(speakers)} speakers):")
        for spk in speakers:
            spk_r = [r for r in ok if r.speaker == spk]
            wer_vals = [r.wer for r in spk_r if r.wer is not None]
            print(f"  {spk:15s}: {len(spk_r):3d} mẫu | WER={fmt(np.mean(wer_vals) if wer_vals else None)}")


# ─────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────

def run_evaluation(
    model,
    model_name: str,
    samples: List[VIVOSSample],
    enabled_metrics: List[str],
    save_audio: bool,
    audio_out_dir: str,
    cache: Dict[str, SampleResult],
    all_results: List[SampleResult],
    cache_path: str,
    save_interval: int = 10,
) -> None:
    import librosa
    import soundfile as sf
    from tqdm import tqdm

    if save_audio:
        os.makedirs(audio_out_dir, exist_ok=True)

    pending = []
    skipped = 0
    for s in samples:
        key = f"{model_name}::{s.sample_id}"
        if key in cache:
            all_results.append(cache[key])
            skipped += 1
        else:
            pending.append(s)

    if skipped:
        logger.info(f"Resume: bỏ qua {skipped} mẫu đã tính, còn {len(pending)} mẫu")

    for idx, sample in enumerate(tqdm(pending, desc=f"[{model_name}]")):
        t0 = time.time()
        result = SampleResult(
            sample_id=sample.sample_id,
            speaker=sample.speaker,
            split=sample.split,
            text=sample.text,
            model_name=model_name,
            success=False,
        )

        try:
            # Tổng hợp: dùng chính file wav của mẫu làm ref (zero-shot clone)
            syn_wav, syn_sr = model.synthesize(
                gen_text=sample.text,
                ref_audio_path=sample.wav_path,
                ref_text=sample.text,
            )
            synth_time = time.time() - t0

            # Tính duration audio tổng hợp
            audio_dur = len(syn_wav) / syn_sr if syn_sr > 0 else 0.0
            result.synth_time_s = round(synth_time, 3)
            result.audio_duration_s = round(audio_dur, 3)
            result.rtf = round(synth_time / audio_dur, 4) if audio_dur > 0 else 0.0
            result.success = True

            # Lưu audio nếu cần
            if save_audio:
                out_wav = os.path.join(audio_out_dir, f"{sample.sample_id}.wav")
                sf.write(out_wav, syn_wav, syn_sr)

            # Tải ref audio để tính metric
            ref_wav, ref_sr = librosa.load(sample.wav_path, sr=None)

            metrics = compute_metrics(
                ref_wav=ref_wav,
                syn_wav=syn_wav,
                ref_sr=ref_sr,
                syn_sr=syn_sr,
                text=sample.text,
                enabled_metrics=enabled_metrics,
            )
            result.mcd = metrics.get("mcd")
            result.pesq = metrics.get("pesq")
            result.stoi = metrics.get("stoi")
            result.utmos = metrics.get("utmos")
            result.f0_corr = metrics.get("f0_corr")
            result.wer = metrics.get("wer")
            result.cer = metrics.get("cer")

        except Exception as e:
            result.error = str(e)
            logger.error(f"Lỗi mẫu {sample.sample_id}: {e}")

        all_results.append(result)

        # Lưu cache định kỳ
        if (idx + 1) % save_interval == 0:
            save_resume_cache(all_results, cache_path)
            logger.debug(f"Cache saved ({idx+1} mẫu)")

    # Lưu cache cuối
    save_resume_cache(all_results, cache_path)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Test F5-TTS finetuned trên VIVOS Vietnamese Speech Corpus"
    )
    p.add_argument("--vivos-dir", default=None,
                   help="Đường dẫn thư mục VIVOS (tự động tìm nếu không chỉ định)")
    p.add_argument("--split", default="test", choices=["train", "test", "both"],
                   help="Split VIVOS cần test (default: test)")
    p.add_argument("--model", default="finetuned",
                   choices=["finetuned", "v0", "both"],
                   help="Model cần test (default: finetuned)")
    p.add_argument("--num-samples", type=int, default=None,
                   help="Giới hạn số mẫu để chạy nhanh (None = tất cả)")
    p.add_argument("--metrics", default="mcd,pesq,stoi,utmos,wer,cer",
                   help="Danh sách metrics ngăn bởi dấu phẩy")
    p.add_argument("--save-audio", action="store_true",
                   help="Lưu file audio tổng hợp")
    p.add_argument("--no-resume", action="store_true",
                   help="Bỏ qua cache, chạy lại từ đầu")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Tốc độ đọc (default: 1.0)")
    p.add_argument("--finetuned-ckpt", default=None,
                   help="Đường dẫn checkpoint finetuned thủ công")
    p.add_argument("--output-dir", default=OUTPUT_DIR,
                   help=f"Thư mục lưu kết quả (default: {OUTPUT_DIR})")
    return p.parse_args()


def main():
    args = parse_args()
    enabled_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    # 1. Tìm VIVOS
    vivos_dir = find_vivos_dir(args.vivos_dir)
    if vivos_dir is None:
        logger.error(
            "Không tìm thấy dataset VIVOS!\n"
            "Hãy chỉ định đường dẫn bằng --vivos-dir hoặc đặt vào một trong:\n"
            + "\n".join(f"  - {d}" for d in DEFAULT_VIVOS_DIRS)
        )
        sys.exit(1)
    logger.info(f"VIVOS dataset: {vivos_dir}")

    # 2. Load samples
    splits = ["train", "test"] if args.split == "both" else [args.split]
    samples = load_vivos(vivos_dir, splits)
    if not samples:
        logger.error("Không có mẫu nào được tải từ VIVOS. Kiểm tra lại đường dẫn dataset.")
        sys.exit(1)

    if args.num_samples:
        samples = samples[: args.num_samples]
        logger.info(f"Giới hạn {args.num_samples} mẫu để test nhanh")

    logger.info(f"Tổng: {len(samples)} mẫu từ {len(set(s.speaker for s in samples))} speakers")

    # 3. Tạo thư mục output
    os.makedirs(args.output_dir, exist_ok=True)
    cache_path = os.path.join(args.output_dir, "resume_cache.json")

    # 4. Load resume cache
    cache = {} if args.no_resume else load_resume_cache(cache_path)
    all_results: List[SampleResult] = []

    # 5. Khởi tạo và chạy model
    models_to_test = []
    if args.model in ("v0", "both"):
        models_to_test.append("v0")
    if args.model in ("finetuned", "both"):
        models_to_test.append("finetuned")

    for model_name in models_to_test:
        logger.info(f"\n{'─'*50}")
        logger.info(f"Đang khởi tạo model: {model_name.upper()}")
        logger.info(f"{'─'*50}")

        try:
            if model_name == "v0":
                from components.tts.F5_V0 import F5TTSVietnamese
                model = F5TTSVietnamese(vocoder_name="vocos", speed=args.speed)
            else:
                from components.tts.F5_Finetuned import F5TTSFinetuned
                model = F5TTSFinetuned(
                    finetuned_ckpt=args.finetuned_ckpt,
                    vocoder_name="vocos",
                    speed=args.speed,
                )
        except Exception as e:
            logger.error(f"Không thể khởi tạo model {model_name}: {e}")
            continue

        audio_out_dir = os.path.join(args.output_dir, "audio", model_name)

        run_evaluation(
            model=model,
            model_name=model_name,
            samples=samples,
            enabled_metrics=enabled_metrics,
            save_audio=args.save_audio,
            audio_out_dir=audio_out_dir,
            cache=cache,
            all_results=all_results,
            cache_path=cache_path,
        )

        # In kết quả ngay sau mỗi model
        print_summary(all_results, model_name)

        # Lưu CSV riêng cho model này
        model_results = [r for r in all_results if r.model_name == model_name]
        csv_path = os.path.join(args.output_dir, f"results_{model_name}.csv")
        save_results_csv(model_results, csv_path)

    # 6. Nếu test cả 2 model, in so sánh
    if len(models_to_test) == 2 and all_results:
        _print_comparison(all_results, models_to_test[0], models_to_test[1])

    # 7. Lưu tất cả kết quả vào một CSV tổng
    all_csv = os.path.join(args.output_dir, "results_all.csv")
    save_results_csv(all_results, all_csv)
    logger.info(f"\nHoàn thành! Kết quả lưu tại: {args.output_dir}")


def _print_comparison(results: List[SampleResult], m1: str, m2: str) -> None:
    def avg(model_name, attr):
        vals = [getattr(r, attr) for r in results
                if r.model_name == model_name and r.success and getattr(r, attr) is not None]
        return np.mean(vals) if vals else None

    def diff_str(v1, v2, lower_better=True):
        if v1 is None or v2 is None:
            return "N/A"
        d = v2 - v1
        if lower_better:
            arrow = "✓ Tốt hơn" if d < 0 else ("✗ Kém hơn" if d > 0 else "=")
        else:
            arrow = "✓ Tốt hơn" if d > 0 else ("✗ Kém hơn" if d < 0 else "=")
        return f"{d:+.4f} ({arrow})"

    print(f"\n{'='*70}")
    print(f"SO SÁNH: {m1.upper()} vs {m2.upper()}")
    print(f"{'='*70}")
    fmt = "{:<12} {:>12} {:>12} {:>25}"
    print(fmt.format("Metric", m1.upper(), m2.upper(), f"{m2.upper()} vs {m1.upper()}"))
    print("-" * 70)
    for metric, lb in [("mcd", True), ("pesq", False), ("stoi", False),
                        ("utmos", False), ("f0_corr", False), ("wer", True), ("cer", True)]:
        v1, v2 = avg(m1, metric), avg(m2, metric)
        f1 = f"{v1:.4f}" if v1 is not None else "N/A"
        f2 = f"{v2:.4f}" if v2 is not None else "N/A"
        print(fmt.format(metric.upper(), f1, f2, diff_str(v1, v2, lb)))
    print("=" * 70)


if __name__ == "__main__":
    main()
