from __future__ import annotations

import argparse
import json
import os
import shutil
from importlib.resources import files
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


def resolve_data_dir(dataset_name: str | None, data_dir: str | None) -> Path:
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    if dataset_name is None:
        raise ValueError("Either --dataset-name or --data-dir must be provided.")
    return Path(files("custom_tts").joinpath(f"../../data/{dataset_name}")).resolve()


def load_raw_dataset(data_dir: Path):
    raw_dir = data_dir / "raw"
    raw_arrow = data_dir / "raw.arrow"
    if raw_dir.exists():
        return load_from_disk(raw_dir.as_posix()), raw_dir
    if raw_arrow.exists():
        return Dataset_.from_file(raw_arrow.as_posix()), raw_arrow
    raise FileNotFoundError(f"Expected {raw_dir} or {raw_arrow}")


def load_audio_16k(audio_path: str, target_sample_rate: int = 16_000) -> torch.Tensor:
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
    return wav.squeeze(0)


def pad_wavs(wavs: list[torch.Tensor]):
    lengths = torch.tensor([wav.numel() for wav in wavs], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.stack([F.pad(wav, (0, max_len - wav.numel())) for wav in wavs])
    attention_mask = torch.arange(max_len)[None, :] < lengths[:, None]
    return padded, attention_mask.long()


@torch.no_grad()
def extract_semantic_batch(
    model,
    feature_extractor,
    audio_paths: list[str],
    device: torch.device,
    layer: int,
    dtype: torch.dtype,
):
    wavs = [load_audio_16k(path) for path in audio_paths]
    padded, attention_mask = pad_wavs(wavs)

    # HuBERT/WavLM feature extractors normalize waveform values and keep masks aligned.
    inputs = feature_extractor(
        padded.numpy(),
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_values = inputs["input_values"].to(device=device, dtype=dtype)
    model_attention_mask = inputs.get("attention_mask", attention_mask).to(device=device)

    outputs = model(input_values, attention_mask=model_attention_mask, output_hidden_states=True)
    hidden = outputs.hidden_states[layer].float().cpu()

    semantic = []
    for item, wav in zip(hidden, wavs):
        # Most HuBERT/WavLM checkpoints stride by about 320 samples at 16 kHz.
        # Trim padded frames conservatively; downstream training interpolates to mel length.
        valid_frames = max(1, int(round(wav.numel() / 320)))
        semantic.append(item[: min(valid_frames, item.shape[0])].half().tolist())
    return semantic


def copy_sidecar_files(src_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("duration.json", "vocab.txt"):
        src = src_dir / name
        dst = out_dir / name
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)


def write_semantic_dataset(
    data_dir: Path,
    out_dir: Path,
    model_name: str,
    layer: int,
    batch_size: int,
    device: str,
    dtype_name: str,
    overwrite: bool,
):
    dataset, raw_source = load_raw_dataset(data_dir)
    if "audio_path" not in dataset.column_names:
        raise ValueError("raw dataset must contain an 'audio_path' column.")

    out_dir.mkdir(parents=True, exist_ok=True)
    copy_sidecar_files(data_dir, out_dir)

    output_arrow = out_dir / "raw.arrow"
    same_arrow = raw_source.resolve() == output_arrow.resolve() if raw_source.exists() else False
    same_raw_dir = raw_source.is_dir() and data_dir.resolve() == out_dir.resolve()
    if (same_arrow or same_raw_dir) and not overwrite:
        raise ValueError(
            f"Refusing to overwrite dataset in {out_dir}. Pass --overwrite or choose a different --output-dir."
        )

    tmp_arrow = output_arrow.with_suffix(".semantic.tmp.arrow")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]
    if torch_device.type == "cpu":
        dtype = torch.float32

    print(f"Loading semantic teacher: {model_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(torch_device)
    if torch_device.type != "cpu":
        model = model.to(dtype=dtype)

    with ArrowWriter(path=tmp_arrow.as_posix(), writer_batch_size=16) as writer:
        for start in tqdm(range(0, len(dataset), batch_size), desc="Extracting HuBERT semantics"):
            end = min(start + batch_size, len(dataset))
            rows = [dict(dataset[i]) for i in range(start, end)]
            semantics = extract_semantic_batch(
                model,
                feature_extractor,
                [row["audio_path"] for row in rows],
                torch_device,
                layer,
                dtype,
            )
            for row, semantic in zip(rows, semantics):
                row["semantic"] = semantic
                writer.write(row)

    if output_arrow.exists() and overwrite:
        backup = output_arrow.with_suffix(".before_semantic.arrow")
        shutil.move(output_arrow.as_posix(), backup.as_posix())
        print(f"Backed up previous raw.arrow to {backup}")

    shutil.move(tmp_arrow.as_posix(), output_arrow.as_posix())

    if same_raw_dir:
        backup_dir = out_dir / "raw.before_semantic"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.move(raw_source.as_posix(), backup_dir.as_posix())
        print(f"Backed up previous raw/ dataset to {backup_dir}")

    print(f"Wrote semantic dataset to {output_arrow}")

    duration_path = out_dir / "duration.json"
    if not duration_path.exists() and "duration" in dataset.column_names:
        with open(duration_path, "w", encoding="utf-8") as f:
            json.dump({"duration": list(dataset["duration"])}, f, ensure_ascii=False)


def cli():
    parser = argparse.ArgumentParser(description="Add HuBERT/WavLM semantic ground truth to a prepared dataset.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset under Custom_TTS/data/<name>.")
    parser.add_argument("--data-dir", type=str, default=None, help="Prepared dataset directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dataset directory. Defaults to input dir.")
    parser.add_argument("--model-name", type=str, default="facebook/hubert-base-ls960", help="HuBERT/WavLM HF model.")
    parser.add_argument("--layer", type=int, default=-1, help="Hidden-state layer to use as semantic GT.")
    parser.add_argument("--batch-size", type=int, default=4, help="Audio batch size for teacher extraction.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Teacher inference dtype on GPU. CPU always uses float32.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing raw.arrow in-place.")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.dataset_name, args.data_dir)
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir
    write_semantic_dataset(
        data_dir=data_dir,
        out_dir=out_dir,
        model_name=args.model_name,
        layer=args.layer,
        batch_size=args.batch_size,
        device=args.device,
        dtype_name=args.dtype,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    cli()
