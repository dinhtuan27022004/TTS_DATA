from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from custom_tts.model.utils import exists, list_str_to_idx, list_str_to_tensor


class TextToSemantic(nn.Module):
    """Predict frame-level semantic features from text tokens.

    The output is intentionally continuous, so it can be supervised by
    HuBERT/WavLM hidden states or used directly as an extra DiT condition.
    """

    def __init__(
        self,
        text_num_embeds: int,
        semantic_dim: int = 768,
        hidden_dim: int = 512,
        depth: int = 4,
        heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj = nn.Linear(hidden_dim, semantic_dim)
        self.semantic_dim = semantic_dim

    def forward(self, text: torch.Tensor, target_len: int | None = None):
        padding_mask = text == -1
        text = text.clamp(min=-1) + 1
        x = self.text_embed(text)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.proj(x)

        if target_len is not None and x.shape[1] != target_len:
            x = F.interpolate(x.transpose(1, 2), size=target_len, mode="nearest").transpose(1, 2)

        return x


class SemanticTeacher(nn.Module):
    """Optional HuBERT/WavLM feature extractor for offline target creation."""

    def __init__(self, model_name: str = "microsoft/wavlm-base-plus", layer: int = -1):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.layer = layer

        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, attention_mask: torch.Tensor | None = None):
        outputs = self.model(wav, attention_mask=attention_mask, output_hidden_states=True)
        return outputs.hidden_states[self.layer]


class SemanticCFM(nn.Module):
    """F5 CFM with an auxiliary text-to-semantic branch.

    This implements the "way 2" design:
    text embedding remains in DiT, while semantic prediction is supplied as an
    additional condition. If semantic_gt is absent, the model still trains with
    normal flow matching and learns the semantic branch through acoustic loss.
    """

    def __init__(
        self,
        cfm: nn.Module,
        semantic_student: TextToSemantic,
        vocab_char_map: dict[str, int] | None = None,
        semantic_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.cfm = cfm
        self.semantic_student = semantic_student
        self.vocab_char_map = vocab_char_map
        self.semantic_loss_weight = semantic_loss_weight
        self.last_losses: dict[str, float] = {}

    @property
    def mel_spec(self):
        return self.cfm.mel_spec

    @property
    def device(self):
        return self.cfm.device

    def _text_to_tokens(self, text):
        if torch.is_tensor(text):
            return text.to(self.device)
        if exists(self.vocab_char_map):
            return list_str_to_idx(text, self.vocab_char_map).to(self.device)
        return list_str_to_tensor(text).to(self.device)

    def predict_semantic(self, text, target_len: int | None = None):
        tokens = self._text_to_tokens(text)
        return self.semantic_student(tokens, target_len=target_len)

    def forward(self, inp, text, *, lens=None, noise_scheduler=None, semantic_gt=None):
        target_len = inp.shape[1] if inp.ndim == 3 else None
        semantic_pred = self.predict_semantic(text, target_len=target_len)

        flow_loss, cond, pred = self.cfm(
            inp,
            text=text,
            lens=lens,
            noise_scheduler=noise_scheduler,
            semantic=semantic_pred,
        )

        loss = flow_loss
        sem_loss = None
        if semantic_gt is not None:
            semantic_gt = semantic_gt.to(device=semantic_pred.device, dtype=semantic_pred.dtype)
            if semantic_gt.shape[1] != semantic_pred.shape[1]:
                semantic_gt = F.interpolate(
                    semantic_gt.transpose(1, 2),
                    size=semantic_pred.shape[1],
                    mode="nearest",
                ).transpose(1, 2)
            sem_loss = F.mse_loss(semantic_pred, semantic_gt)
            loss = loss + self.semantic_loss_weight * sem_loss

        self.last_losses = {
            "flow": float(flow_loss.detach().cpu()),
            "semantic": float(sem_loss.detach().cpu()) if sem_loss is not None else 0.0,
            "total": float(loss.detach().cpu()),
        }
        return loss, cond, pred

    @torch.no_grad()
    def sample(self, cond, text, duration, **kwargs):
        if isinstance(duration, int):
            target_len = duration
        elif torch.is_tensor(duration):
            target_len = int(duration.max().item())
        else:
            target_len = None

        semantic = self.predict_semantic(text, target_len=target_len)
        return self.cfm.sample(cond=cond, text=text, duration=duration, semantic=semantic, **kwargs)


def collate_semantic_targets(batch):
    semantic = [item.get("semantic") for item in batch]
    if not semantic or any(item is None for item in semantic):
        return None

    tensors = [torch.as_tensor(item, dtype=torch.float32) for item in semantic]
    return pad_sequence(tensors, batch_first=True, padding_value=0.0)
