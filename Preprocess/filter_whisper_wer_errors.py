#!/usr/bin/env python3
"""
Filter WAV/TXT pairs whose Whisper transcript differs from the reference text.

For each .wav file in a dataset directory, this script expects a .txt file with
the same stem in the same folder. It transcribes the wav with faster-whisper,
computes WER, and moves pairs with WER > threshold to data/Error.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import time
from tqdm import tqdm
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "YTDT2"
DEFAULT_ERROR_DIR = PROJECT_ROOT / "data" / "Error"


for noisy_logger in ("faster_whisper", "faster_whisper.transcribe", "ctranslate2"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Whisper on a dataset of paired .wav/.txt files, then move pairs "
            "with non-zero WER to data/Error and log details to JSON."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        nargs="?",
        default=DATASET_DIR,
        help="Path to dataset folder containing paired .wav/.txt files.",
    )
    parser.add_argument(
        "--error-dir",
        type=Path,
        default=DEFAULT_ERROR_DIR,
        help=f"Directory to store moved error pairs. Default: {DEFAULT_ERROR_DIR}",
    )
    parser.add_argument(
        "--json-name",
        default="wer_errors.json",
        help="JSON filename written inside --error-dir. Default: wer_errors.json",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="faster-whisper model name/path. Default: large-v3",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu", "auto"),
        help="Device for faster-whisper. Default: cuda",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="faster-whisper compute type. Use int8 for lower VRAM. Default: float16",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of faster-whisper workers. Default: 4",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Whisper beam size. Default: 5",
    )
    parser.add_argument(
        "--language",
        default="vi",
        help="Whisper language code. Default: vi",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Move files when WER is greater than this value. Default: 0.0",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N wav files. Useful for testing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and log errors but do not move files.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing JSON report and transcribe from the beginning.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    previous = list(range(len(hyp_words) + 1))
    for i, ref_word in enumerate(ref_words, start=1):
        current = [i]
        for j, hyp_words_loop in enumerate(hyp_words, start=1):
            substitution_cost = 0 if ref_word == hyp_words_loop else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current

    return previous[-1] / len(ref_words)


def load_whisper_worker(args: argparse.Namespace):
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        num_workers=args.num_workers
    )
    return model


def transcribe(worker, wav_path: Path, args: argparse.Namespace) -> str:
    segments, _ = worker.transcribe(str(wav_path), beam_size=args.beam_size, language=args.language)
    text_content = " ".join([seg.text.strip() for seg in segments]).strip()
    return text_content


def collect_pairs(dataset_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
    wav_files = sorted(dataset_dir.rglob("*.wav"))
    pairs: List[Tuple[Path, Path]] = []
    missing_txt: List[Path] = []

    for wav_path in wav_files:
        txt_path = wav_path.with_suffix(".txt")
        if txt_path.exists():
            pairs.append((wav_path, txt_path))
        else:
            missing_txt.append(wav_path)

    return pairs, missing_txt


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 2
    while True:
        candidate = parent / f"{stem}__dup{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def move_pair(
    wav_path: Path,
    txt_path: Path,
    dataset_dir: Path,
    error_dataset_dir: Path,
) -> Tuple[Path, Path]:
    rel_parent = wav_path.parent.relative_to(dataset_dir)
    dest_dir = error_dataset_dir / rel_parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_wav = unique_destination(dest_dir / wav_path.name)
    dest_txt = dest_wav.with_suffix(".txt")
    if dest_txt.exists():
        dest_txt = unique_destination(dest_dir / txt_path.name)

    shutil.move(str(wav_path), str(dest_wav))
    shutil.move(str(txt_path), str(dest_txt))
    return dest_wav, dest_txt


def save_report(report_path: Path, report: Dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(report_path)


def load_existing_report(report_path: Path) -> Optional[Dict]:
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Cannot read existing report, starting fresh: {report_path} ({exc})")
        return None


def processed_key(path: Path, dataset_dir: Path) -> str:
    return path.relative_to(dataset_dir).as_posix()


def build_processed_set(report: Dict, dataset_dir: Path) -> set[str]:
    processed = set(report.get("processed_files", []))

    # Backward compatibility for reports created before processed_files existed.
    for item in report.get("errors", []):
        original_wav = item.get("original_wav")
        if original_wav:
            try:
                processed.add(Path(original_wav).resolve().relative_to(dataset_dir).as_posix())
            except ValueError:
                processed.add(str(original_wav))

    for item in report.get("transcribe_errors", []):
        wav_file = item.get("wav_file")
        if wav_file:
            try:
                processed.add(Path(wav_file).resolve().relative_to(dataset_dir).as_posix())
            except ValueError:
                processed.add(str(wav_file))

    return processed


def make_new_report(
    args: argparse.Namespace,
    dataset_dir: Path,
    error_dir: Path,
    error_dataset_dir: Path,
    pairs: List[Tuple[Path, Path]],
    missing_txt: List[Path],
) -> Dict:
    return {
        "dataset_dir": str(dataset_dir),
        "error_dir": str(error_dir),
        "error_dataset_dir": str(error_dataset_dir),
        "model": args.model,
        "language": args.language,
        "threshold": args.threshold,
        "dry_run": args.dry_run,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_resumed_at": None,
        "finished_at": None,
        "total_pairs": len(pairs),
        "missing_txt_count": len(missing_txt),
        "missing_txt": [str(path) for path in missing_txt],
        "processed": 0,
        "processed_files": [],
        "ok_count": 0,
        "error_count": 0,
        "transcribe_error_count": 0,
        "errors": [],
        "transcribe_errors": [],
    }


def prepare_report(
    args: argparse.Namespace,
    report_path: Path,
    dataset_dir: Path,
    error_dir: Path,
    error_dataset_dir: Path,
    pairs: List[Tuple[Path, Path]],
    missing_txt: List[Path],
) -> Tuple[Dict, set[str]]:
    if args.no_resume:
        report = make_new_report(args, dataset_dir, error_dir, error_dataset_dir, pairs, missing_txt)
        return report, set()

    existing_report = load_existing_report(report_path)
    if existing_report is None:
        report = make_new_report(args, dataset_dir, error_dir, error_dataset_dir, pairs, missing_txt)
        return report, set()

    report = existing_report
    processed = build_processed_set(report, dataset_dir)
    report.setdefault("processed_files", sorted(processed))
    report.setdefault("ok_count", max(0, int(report.get("processed", 0)) - int(report.get("error_count", 0))))
    report.setdefault("errors", [])
    report.setdefault("transcribe_errors", [])
    report["dataset_dir"] = str(dataset_dir)
    report["error_dir"] = str(error_dir)
    report["error_dataset_dir"] = str(error_dataset_dir)
    report["model"] = args.model
    report["language"] = args.language
    report["threshold"] = args.threshold
    report["dry_run"] = args.dry_run
    report["last_resumed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["finished_at"] = None
    current_keys = {processed_key(wav_path, dataset_dir) for wav_path, _ in pairs}
    moved_or_missing_processed = sum(1 for key in processed if key not in current_keys)
    report["total_pairs"] = max(
        int(report.get("total_pairs", 0)),
        len(pairs) + moved_or_missing_processed,
    )
    report["missing_txt_count"] = len(missing_txt)
    report["missing_txt"] = [str(path) for path in missing_txt]
    report["processed"] = len(processed)
    report["error_count"] = len(report.get("errors", []))
    report["transcribe_error_count"] = len(report.get("transcribe_errors", []))
    return report, processed


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    error_dir = args.error_dir.expanduser().resolve()
    error_dataset_dir = error_dir / dataset_dir.name
    report_path = error_dir / args.json_name

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(f"Dataset directory does not exist: {dataset_dir}")
        return 1

    pairs, missing_txt = collect_pairs(dataset_dir)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    if not pairs:
        print(f"No paired .wav/.txt files found in {dataset_dir}")
        return 1

    report, processed_files = prepare_report(
        args,
        report_path,
        dataset_dir,
        error_dir,
        error_dataset_dir,
        pairs,
        missing_txt,
    )
    pairs_to_process = [
        (wav_path, txt_path)
        for wav_path, txt_path in pairs
        if processed_key(wav_path, dataset_dir) not in processed_files
    ]

    print(f"Dataset: {dataset_dir}")
    print(f"Pairs: {len(pairs):,} | missing txt: {len(missing_txt):,}")
    print(f"Already processed: {len(processed_files):,}")
    print(f"Remaining: {len(pairs_to_process):,}")
    print(f"Error output: {error_dataset_dir}")
    print(f"Report: {report_path}")
    if not pairs_to_process:
        report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_report(report_path, report)
        print("Nothing to process. Resume state is already complete.")
        return 0

    print("Loading Whisper...")
    worker = load_whisper_worker(args)
    print("Whisper loaded. Processing...")

    save_report(report_path, report)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(transcribe, worker, wav_path, args): (wav_path, txt_path)
            for wav_path, txt_path in pairs_to_process
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Whisper WER"):
            wav_path, txt_path = futures[future]
            key = processed_key(wav_path, dataset_dir)
            raw_text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
            
            try:
                gen_text = future.result()
            except Exception as exc:
                report["transcribe_error_count"] += 1
                report["transcribe_errors"].append(
                    {
                        "wav_file": str(wav_path),
                        "txt_file": str(txt_path),
                        "error": str(exc),
                    }
                )
                processed_files.add(key)
                report["processed_files"] = sorted(processed_files)
                report["processed"] += 1
                save_report(report_path, report)
                continue

            ref_norm = normalize_text(raw_text)
            hyp_norm = normalize_text(gen_text)
            wer_score = word_error_rate(ref_norm, hyp_norm)

            if wer_score > args.threshold:
                moved_wav: Optional[Path] = None
                moved_txt: Optional[Path] = None
                if not args.dry_run:
                    moved_wav, moved_txt = move_pair(
                        wav_path,
                        txt_path,
                        dataset_dir,
                        error_dataset_dir,
                    )

                report["error_count"] += 1
                report["errors"].append(
                    {
                        "file_name": wav_path.name,
                        "stem": wav_path.stem,
                        "wer": wer_score,
                        "wer_percent": wer_score * 100.0,
                        "raw_text": raw_text,
                        "gen_text": gen_text,
                        "raw_text_normalized": ref_norm,
                        "gen_text_normalized": hyp_norm,
                        "original_wav": str(wav_path),
                        "original_txt": str(txt_path),
                        "moved_wav": str(moved_wav) if moved_wav else None,
                        "moved_txt": str(moved_txt) if moved_txt else None,
                    }
                )
                save_report(report_path, report)
            else:
                report["ok_count"] = report.get("ok_count", 0) + 1

            processed_files.add(key)
            report["processed_files"] = sorted(processed_files)
            report["processed"] += 1
            if report["processed"] % 100 == 0:
                save_report(report_path, report)

    report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_report(report_path, report)

    print("\nDone")
    print(f"Processed: {report['processed']:,}/{report['total_pairs']:,}")
    print(f"Moved error pairs: {report['error_count']:,}")
    print(f"Transcribe errors: {report['transcribe_error_count']:,}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
