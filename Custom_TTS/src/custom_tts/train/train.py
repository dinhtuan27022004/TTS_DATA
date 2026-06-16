# training script.

import os
import shutil
from importlib.resources import files

import hydra
from cached_path import cached_path
from omegaconf import OmegaConf

from custom_tts.model import CFM, DiT, SemanticCFM, TextToSemantic, UNetT, Trainer  # noqa: F401. used for config
from custom_tts.model.dataset import load_dataset
from custom_tts.model.utils import get_tokenizer

os.chdir(str(files("custom_tts").joinpath("../..")))  # change working directory to root of project (local editable)


def prepare_pretrained_checkpoint(checkpoint_path: str, pretrained_cfg) -> str | None:
    if not pretrained_cfg or not pretrained_cfg.get("enabled", False):
        return None

    os.makedirs(checkpoint_path, exist_ok=True)

    has_training_checkpoint = any(
        name == "model_last.pt" or (name.startswith("model_") and name.endswith((".pt", ".safetensors")))
        for name in os.listdir(checkpoint_path)
    )
    if has_training_checkpoint:
        return None

    source = pretrained_cfg.get("path") or pretrained_cfg.get("url")
    if not source:
        return None

    source_path = str(cached_path(source)) if source.startswith(("hf://", "http://", "https://")) else source
    filename = os.path.basename(source_path)
    if not filename.startswith("pretrained_"):
        filename = f"pretrained_{filename}"

    target_path = os.path.join(checkpoint_path, filename)
    if not os.path.isfile(target_path):
        shutil.copy2(source_path, target_path)
        print(f"Prepared pretrained checkpoint: {target_path}")
    else:
        print(f"Using existing pretrained checkpoint: {target_path}")

    return target_path


@hydra.main(version_base="1.3", config_path=str(files("custom_tts").joinpath("configs")), config_name=None)
def main(cfg):
    model_cls = globals()[cfg.model.backbone]
    model_arc = cfg.model.arch
    tokenizer = cfg.model.tokenizer
    mel_spec_type = cfg.model.mel_spec.mel_spec_type

    exp_name = f"{cfg.model.name}_{mel_spec_type}_{cfg.model.tokenizer}_{cfg.datasets.name}"
    wandb_resume_id = None

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = cfg.datasets.name
    else:
        tokenizer_path = cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    vocab_size_override = cfg.model.get("vocab_size_override")
    if vocab_size_override is not None:
        if vocab_size > vocab_size_override:
            raise ValueError(
                f"vocab_size_override={vocab_size_override} is smaller than tokenizer vocab size {vocab_size}"
            )
        print(f"Using fixed vocab size: tokenizer={vocab_size}, model={vocab_size_override}")
        vocab_size = vocab_size_override

    # set model
    cfm = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )
    if cfg.model.get("semantic", {}).get("enabled", False):
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
        model = SemanticCFM(
            cfm=cfm,
            semantic_student=semantic_student,
            vocab_char_map=vocab_char_map,
            semantic_loss_weight=semantic_cfg.loss_weight,
        )
    else:
        model = cfm

    checkpoint_path = str(files("custom_tts").joinpath(f"../../{cfg.ckpts.save_dir}"))
    prepare_pretrained_checkpoint(checkpoint_path, cfg.ckpts.get("pretrained"))

    # init trainer
    trainer = Trainer(
        model,
        epochs=cfg.optim.epochs,
        learning_rate=cfg.optim.learning_rate,
        num_warmup_updates=cfg.optim.num_warmup_updates,
        save_per_updates=cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=cfg.datasets.batch_size_per_gpu,
        batch_size_type=cfg.datasets.batch_size_type,
        max_samples=cfg.datasets.max_samples,
        grad_accumulation_steps=cfg.optim.grad_accumulation_steps,
        max_grad_norm=cfg.optim.max_grad_norm,
        logger=cfg.ckpts.logger,
        wandb_project="CFM-TTS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=cfg.ckpts.last_per_updates,
        log_samples=cfg.ckpts.log_samples,
        bnb_optimizer=cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=cfg.model.vocoder.is_local,
        local_vocoder_path=cfg.model.vocoder.local_path,
        cfg_dict=OmegaConf.to_container(cfg, resolve=True),
    )

    train_dataset = load_dataset(cfg.datasets.name, tokenizer, mel_spec_kwargs=cfg.model.mel_spec)
    trainer.train(
        train_dataset,
        num_workers=cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
