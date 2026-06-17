from __future__ import annotations

import argparse
import os
from importlib.resources import files
from pathlib import Path

import torch
from cached_path import cached_path
from omegaconf import OmegaConf

from custom_tts.model import CFM, DiT, SemanticCFM, TextToSemantic, UNetT  # noqa: F401
from custom_tts.model.utils import get_tokenizer


def load_plain_state_dict(checkpoint_path: str):
    suffix = checkpoint_path.rsplit(".", 1)[-1]
    if suffix == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(checkpoint_path, device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if "ema_model_state_dict" in checkpoint:
        state_dict = {
            key.replace("ema_model.", ""): value
            for key, value in checkpoint["ema_model_state_dict"].items()
            if key not in ["initted", "update", "step"]
        }
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
        state_dict.pop(key, None)

    return state_dict


def adapt_state_dict_for_model(model, state_dict):
    if hasattr(model, "cfm") and not any(key.startswith("cfm.") for key in state_dict):
        state_dict = {f"cfm.{key}": value for key, value in state_dict.items()}

    target_state = model.state_dict()
    adapted = {}
    skipped = []
    expanded = []

    for key, value in state_dict.items():
        if key not in target_state:
            skipped.append(key)
            continue

        target = target_state[key]
        if value.shape == target.shape:
            adapted[key] = value
            continue

        if value.ndim == target.ndim and all(src <= dst for src, dst in zip(value.shape, target.shape)):
            new_value = target.clone()
            slices = tuple(slice(0, size) for size in value.shape)
            new_value[slices] = value
            adapted[key] = new_value
            expanded.append(key)
        else:
            skipped.append(key)

    return adapted, skipped, expanded


def build_semantic_model(cfg, vocab_size_override: int | None = None):
    model_cls = globals()[cfg.model.backbone]
    model_arc = cfg.model.arch
    tokenizer = cfg.model.tokenizer

    if tokenizer != "custom":
        tokenizer_path = cfg.datasets.name
    else:
        tokenizer_path = cfg.model.tokenizer_path

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    fixed_vocab_size = vocab_size_override or cfg.model.get("vocab_size_override")
    if fixed_vocab_size is not None:
        if vocab_size > fixed_vocab_size:
            raise ValueError(
                f"fixed vocab size {fixed_vocab_size} is smaller than tokenizer vocab size {vocab_size}"
            )
        print(f"Using fixed vocab size: tokenizer={vocab_size}, model={fixed_vocab_size}")
        vocab_size = fixed_vocab_size
    cfm = CFM(
        transformer=model_cls(
            **model_arc,
            text_num_embeds=vocab_size,
            mel_dim=cfg.model.mel_spec.n_mel_channels,
        ),
        mel_spec_kwargs=cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    semantic_cfg = cfg.model.semantic
    semantic_student = TextToSemantic(
        text_num_embeds=vocab_size,
        semantic_dim=semantic_cfg.dim,
        hidden_dim=semantic_cfg.hidden_dim,
        depth=semantic_cfg.depth,
        heads=semantic_cfg.heads,
        ff_mult=semantic_cfg.ff_mult,
        dropout=semantic_cfg.dropout,
    )
    return SemanticCFM(
        cfm=cfm,
        semantic_student=semantic_student,
        vocab_char_map=vocab_char_map,
        semantic_loss_weight=semantic_cfg.loss_weight,
    )


def resolve_pretrained(pretrained: str | None, cfg):
    source = pretrained
    if source is None:
        pretrained_cfg = cfg.ckpts.get("pretrained")
        if pretrained_cfg:
            source = pretrained_cfg.get("path") or pretrained_cfg.get("url")
    if not source:
        raise ValueError("No pretrained checkpoint provided and cfg.ckpts.pretrained is empty.")
    return str(cached_path(source)) if source.startswith(("hf://", "http://", "https://")) else source


def default_output_path(cfg):
    save_dir = OmegaConf.to_container(OmegaConf.create({"save_dir": cfg.ckpts.save_dir}), resolve=True)["save_dir"]
    ckpt_dir = Path(files("custom_tts").joinpath(f"../../{save_dir}"))
    return ckpt_dir / "pretrained_semantic_init.pt"


def cli():
    parser = argparse.ArgumentParser(description="Initialize a SemanticCFM .pt checkpoint from a plain F5 checkpoint.")
    parser.add_argument("--config-name", type=str, default="SemanticF5TTS_Base.yaml")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None, help="F5 checkpoint path/url. Defaults to config ckpts.pretrained.")
    parser.add_argument("--output", type=str, default=None, help="Output .pt path.")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override model embedding vocab size.")
    args = parser.parse_args()

    config_path = files("custom_tts").joinpath("configs").joinpath(args.config_name)
    cfg = OmegaConf.load(config_path)
    if args.dataset_name:
        cfg.datasets.name = args.dataset_name
    if "hydra" in cfg:
        del cfg["hydra"]

    model = build_semantic_model(cfg, vocab_size_override=args.vocab_size)
    pretrained_path = resolve_pretrained(args.pretrained, cfg)
    f5_state = load_plain_state_dict(pretrained_path)
    adapted, skipped, expanded = adapt_state_dict_for_model(model, f5_state)
    incompatible = model.load_state_dict(adapted, strict=False)

    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_state = model.state_dict()
    ema_model_state = {f"ema_model.{key}": value.cpu() for key, value in model_state.items()}
    torch.save(
        {
            "ema_model_state_dict": ema_model_state,
            "init_from": pretrained_path,
            "semantic_init": True,
        },
        output_path,
    )

    print(f"Saved semantic init checkpoint: {output_path}")
    print(f"Loaded compatible tensors: {len(adapted)}")
    print(f"Expanded tensors: {len(expanded)}")
    print(f"Missing new-model keys: {len(incompatible.missing_keys)}")
    print(f"Unexpected loaded keys: {len(incompatible.unexpected_keys)}")
    if skipped:
        print(f"Skipped incompatible/source-only tensors: {len(skipped)}")


if __name__ == "__main__":
    cli()
