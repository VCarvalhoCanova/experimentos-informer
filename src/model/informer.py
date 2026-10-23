from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

__all__ = [
    "ProbSparseAttention",
    "FullAttention",
    "AttentionLayer",
    "ConvLayer",
    "EncoderLayer",
    "Encoder",
    "DecoderLayer",
    "Decoder",
    "DataEmbedding",
    "Informer",
    "InformerForecaster",
    "InformerMultivariateForecaster",
    "time_features",
]

# ---------------------------------------------------------------------------
# Time features (uniform input representation -- Appendix B)
# ---------------------------------------------------------------------------

#: Column order expected by :class:`TemporalEmbedding`: month, day, weekday,
#: hour, minute. Same order used by the official Informer implementation.
TIME_FEATURE_COLUMNS = ["month", "day", "weekday", "hour", "minute"]


def time_features(dates: Union[pd.DatetimeIndex, pd.Series, None], n: int) -> np.ndarray:
    """Criar calendário para embeddings
    """
    if dates is None:
        dates = pd.date_range("2016-01-01", periods=n, freq="h")

    index = pd.DatetimeIndex(dates)
    if len(index) < n:
        # Extrapolate with the inferred frequency when the caller asks for
        # future marks (decoder placeholder).
        freq = pd.infer_freq(index) if len(index) >= 3 else None
        extra = pd.date_range(index[-1], periods=n - len(index) + 1, freq=freq or "h")[1:]
        index = index.append(extra)

    index = index[:n]
    marks = np.stack(
        [
            index.month.to_numpy() - 1,
            index.day.to_numpy() - 1,
            index.weekday.to_numpy(),
            index.hour.to_numpy(),
            index.minute.to_numpy(),
        ],
        axis=1,
    )
    return marks.astype(np.int64)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class ProbSparseAttention(nn.Module):
   
    def __init__(self, factor: int = 5, attention_dropout: float = 0.1) -> None:

        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(attention_dropout)
        self.scale: Optional[float] = None

    def _prob_qk(
        self, q: torch.Tensor, k: torch.Tensor, sample_k: int, n_top: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selecionar queries dominantes com base na medição de sparsidade.
        """
        batch, heads, len_q, dim = q.shape
        len_k = k.shape[2]

        # K_expand[b, h, i, j, :] = k[b, h, j, :]
        k_expand = k.unsqueeze(-3).expand(batch, heads, len_q, len_k, dim)
        index_sample = torch.randint(len_k, (len_q, sample_k), device=q.device)
        k_sample = k_expand[:, :, torch.arange(len_q, device=q.device).unsqueeze(1), index_sample, :]

        # Sparse scores only for the sampled query-key pairs.
        q_k_sample = torch.matmul(q.unsqueeze(-2), k_sample.transpose(-2, -1)).squeeze(-2)

        # M(q_i, K) = max_j(q_i k_j^T / sqrt(d)) - mean_j(q_i k_j^T / sqrt(d))
        measurement = q_k_sample.max(-1)[0] - q_k_sample.mean(-1)
        index = measurement.topk(n_top, sorted=False)[1]

        q_reduce = q[
            torch.arange(batch, device=q.device)[:, None, None],
            torch.arange(heads, device=q.device)[None, :, None],
            index,
            :,
        ]

        return q_reduce, index

    @staticmethod
    def _gather_mask(
        attn_mask: torch.Tensor, index: torch.Tensor, batch: int, heads: int, len_k: int
    ) -> torch.Tensor:
        """Selecionar mascaras que correspondem aos queries dominantes."""
        if attn_mask.dim() == 2:
            mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.dim() == 3:
            mask = attn_mask.unsqueeze(1)
        else:
            mask = attn_mask

        mask = mask.expand(batch, heads, mask.shape[-2], len_k)
        return mask.gather(2, index.unsqueeze(-1).expand(batch, heads, -1, len_k))

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Executa a atenção ProbSparse multi-cabeça.
        """
        batch, len_q, heads, dim = queries.shape
        len_k = keys.shape[1]

        scale = self.scale or 1.0 / math.sqrt(dim)

        if len_q > len_k:  # pad keys/values (cross attention safety)
            zeros = torch.zeros_like(queries[:, : len_q - len_k, :])
            keys = torch.cat([keys, zeros], dim=1)
            values = torch.cat([values, zeros], dim=1)
        else:
            keys = keys[:, :len_q, :]
            values = values[:, :len_q, :]
        len_k = keys.shape[1]

        q = queries.permute(0, 2, 1, 3)  # (B, H, L_Q, E)
        k = keys.permute(0, 2, 1, 3)
        v = values.permute(0, 2, 1, 3)

        # u = c * ln(L_Q) and U_part = c * ln(L_K), bounded by the sequence size.
        u_part = min(int(self.factor * math.ceil(math.log(len_k))), len_k)
        u = min(int(self.factor * math.ceil(math.log(len_q))), len_q)

        q_reduce, index = self._prob_qk(q, k, sample_k=u_part, n_top=u)

        scores = torch.matmul(q_reduce, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            mask = self._gather_mask(attn_mask, index, batch, heads, len_k)
            scores = scores.masked_fill(mask, -np.inf)

        attn = F.softmax(scores, dim=-1)
        context_reduce = torch.matmul(self.dropout(attn), v)

        context = torch.zeros(batch, heads, len_q, v.shape[-1], device=queries.device, dtype=queries.dtype)
        context.scatter_(
            2,
            index.unsqueeze(-1).expand(batch, heads, u, v.shape[-1]),
            context_reduce,
        )

        return context.permute(0, 2, 1, 3), None


class FullAttention(nn.Module):
    """Atenção completa canonica"""

    def __init__(self, attention_dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.scale: Optional[float] = None

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executa a atenção completa multi-cabeça.
        """
        scale = self.scale or 1.0 / math.sqrt(queries.shape[-1])

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), -np.inf)
            elif attn_mask.dim() == 3:
                scores = scores.masked_fill(attn_mask.unsqueeze(1), -np.inf)
            else:
                scores = scores.masked_fill(attn_mask, -np.inf)

        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.einsum("bhls,bshd->blhd", attn, values)
        return context, attn


class AttentionLayer(nn.Module):

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None,
    ) -> None:
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, len_q, _ = queries.shape
        _, len_k, _ = keys.shape

        queries = self.query_projection(queries).view(batch, len_q, self.n_heads, -1)
        keys = self.key_projection(keys).view(batch, len_k, self.n_heads, -1)
        values = self.value_projection(values).view(batch, len_k, self.n_heads, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.reshape(batch, len_q, -1)
        return self.out_projection(out), attn


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class PositionalEmbedding(nn.Module):
    """Encoding posicional senoidal"""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:

        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (2.0 * max_len) ** (torch.arange(0, d_model, 2).float() / d_model)
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class TemporalEmbedding(nn.Module):
    """Embedding de calendário: mês, dia, dia da semana, hora, minuto"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.month_embed = nn.Embedding(13, d_model)
        self.day_embed = nn.Embedding(32, d_model)
        self.weekday_embed = nn.Embedding(7, d_model)
        self.hour_embed = nn.Embedding(24, d_model)
        self.minute_embed = nn.Embedding(60, d_model)

    def forward(self, marks: torch.Tensor) -> torch.Tensor:
        marks = marks.long()
        return (
            self.month_embed(marks[:, :, 0])
            + self.day_embed(marks[:, :, 1])
            + self.weekday_embed(marks[:, :, 2])
            + self.hour_embed(marks[:, :, 3])
            + self.minute_embed(marks[:, :, 4])
        )


class DataEmbedding(nn.Module):
    """Uniform input representation: value projection + position + calendar."""

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        # alpha * u_i: the scalar is projected with a 1-D convolution (kernel 3).
        self.value_embedding = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1, bias=False)
        self.position_embedding = PositionalEmbedding(d_model, max_len)
        self.temporal_embedding = TemporalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
        x = self.value_embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.dropout(x + self.position_embedding(x) + self.temporal_embedding(marks))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class ConvLayer(nn.Module):

    def __init__(self, c_in: int) -> None:
        super().__init__()
        self.down_conv = nn.Conv1d(c_in, c_in, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        # stride 2 halves the time dimension, as required by the paper.
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_conv(x.permute(0, 2, 1))
        x = self.activation(self.norm(x))
        x = self.max_pool(x)
        return x.permute(0, 2, 1)


class EncoderLayer(nn.Module):
    """One encoder layer: ProbSparse self-attention + two feed-forward convs."""

    def __init__(self, attention: nn.Module, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):

    def __init__(self, attn_layers: List[EncoderLayer], conv_layers: Optional[List[ConvLayer]] = None) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the stack."""
        for i, attn_layer in enumerate(self.attn_layers):
            x, _ = attn_layer(x, attn_mask=attn_mask)
            if self.conv_layers is not None and i < len(self.conv_layers):
                x = self.conv_layers[i](x)
        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class DecoderLayer(nn.Module):

    def __init__(
        self,
        self_attention: nn.Module,
        cross_attention: nn.Module,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0]))
        x = self.norm2(x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]))

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y), None


class Decoder(nn.Module):

    def __init__(self, layers: List[DecoderLayer], norm_layer: nn.Module, projection: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x, _ = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        return self.projection(self.norm(x))


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------


class Informer(nn.Module):

    def __init__(
        self,
        seq_len: int,
        label_len: int,
        pred_len: int,
        c_in: int = 1,
        c_out: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 128,
        factor: int = 5,
        dropout: float = 0.1,
        distil: bool = True,
        n_encoder_stacks: int = 2,
        replica_layers: Optional[int] = None,
        replica_input_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < replica_input_fraction <= 1.0:
            raise ValueError("replica_input_fraction must be in (0, 1]")

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.n_encoder_stacks = max(1, int(n_encoder_stacks))
        self.replica_layers = replica_layers
        self.replica_input_fraction = replica_input_fraction

        self.enc_embedding = DataEmbedding(c_in, d_model, dropout, max_len=seq_len + pred_len)
        self.dec_embedding = DataEmbedding(c_in, d_model, dropout, max_len=seq_len + pred_len)

        # Pyramid (Fig. 3): the main stack sees the whole input; each replica
        # sees a fixed fraction of it and has fewer layers. Defaults reproduce
        # the original behaviour (halved input, one layer less per replica).
        if replica_layers is None:
            stack_layers = [max(1, e_layers - stack) for stack in range(self.n_encoder_stacks)]
        else:
            stack_layers = [max(1, e_layers)] + [max(1, int(replica_layers))] * (self.n_encoder_stacks - 1)
        self.encoder_input_fractions = [1.0] + [
            replica_input_fraction ** stack for stack in range(1, self.n_encoder_stacks)
        ]

        encoders = []
        for n_layers in stack_layers:
            layers = [
                EncoderLayer(
                    AttentionLayer(ProbSparseAttention(factor, dropout), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout,
                )
                for _ in range(n_layers)
            ]
            convs = (
                [ConvLayer(d_model) for _ in range(n_layers - 1)]
                if (distil and n_layers > 1)
                else None
            )
            encoders.append(Encoder(layers, convs))
        self.encoders = nn.ModuleList(encoders)

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(ProbSparseAttention(factor, dropout), d_model, n_heads),
                    AttentionLayer(FullAttention(dropout), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout,
                )
                for _ in range(d_layers)
            ],
            nn.LayerNorm(d_model),
            nn.Linear(d_model, c_out, bias=True),
        )

    def forward(
        self,
        enc_x: torch.Tensor,
        enc_marks: torch.Tensor,
        dec_x: torch.Tensor,
        dec_marks: torch.Tensor,
    ) -> torch.Tensor:
        enc_out = self.enc_embedding(enc_x, enc_marks)

        # Encoder replicas: each stack receives the last fraction of the input
        # (the main stack receives all of it) so the outputs can be concatenated.
        stack_outputs = []
        input_len = enc_out.shape[1]
        for fraction, encoder in zip(self.encoder_input_fractions, self.encoders):
            keep = max(1, int(round(input_len * fraction)))
            out = encoder(enc_out[:, input_len - keep :, :])
            stack_outputs.append(out)

        min_len = min(out.shape[1] for out in stack_outputs)
        enc_feature_map = torch.cat([out[:, -min_len:, :] for out in stack_outputs], dim=1)

        dec_out = self.dec_embedding(dec_x, dec_marks)

        # Causal mask for the masked ProbSparse self-attention of the decoder.
        dec_len = dec_out.shape[1]
        mask = torch.triu(torch.ones(dec_len, dec_len, device=dec_out.device, dtype=torch.bool), diagonal=1)

        out = self.decoder(dec_out, enc_feature_map, x_mask=mask)
        return out[:, -self.pred_len :, :]


# ---------------------------------------------------------------------------
# Scikit-learn style wrapper used by the demonstration notebook
# ---------------------------------------------------------------------------


class InformerForecaster:

    def __init__(
        self,
        seq_len: int = 96,
        label_len: Optional[int] = None,
        pred_len: int = 24,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 128,
        factor: int = 5,
        dropout: float = 0.1,
        distil: bool = True,
        n_encoder_stacks: int = 2,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 3,
        val_ratio: float = 0.1,
        seed: int = 42,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        self.seq_len = int(seq_len)
        self.label_len = int(label_len if label_len is not None else max(1, seq_len // 2))
        self.pred_len = int(pred_len)
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.factor = factor
        self.dropout = dropout
        self.distil = distil
        self.n_encoder_stacks = n_encoder_stacks
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_ratio = val_ratio
        self.seed = seed
        self.device = torch.device(device)
        self.verbose = verbose

        self.model: Optional[Informer] = None
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.time_exec: Dict[str, Optional[float]] = {"training": None, "testing": None}
        self.mean_: float = 0.0
        self.std_: float = 1.0

    # -- helpers ---------------------------------------------------------
    @property
    def total_input_len(self) -> int:
        return self.seq_len + self.pred_len

    def _standardize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_) / self.std_

    def _destandardize(self, values: np.ndarray) -> np.ndarray:
        return values * self.std_ + self.mean_

    def _windows(
        self,
        values: np.ndarray,
        marks: np.ndarray,
        origins: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_x, dec_x, enc_m, dec_m, target = [], [], [], [], []
        for origin in origins:
            enc_end = origin + self.seq_len
            dec_end = enc_end + self.pred_len

            enc_x.append(values[origin:enc_end])
            dec_values = np.zeros(self.label_len + self.pred_len, dtype=np.float32)
            dec_values[: self.label_len] = values[enc_end - self.label_len : enc_end]
            dec_x.append(dec_values)

            enc_m.append(marks[origin:enc_end])
            dec_m.append(marks[enc_end - self.label_len : dec_end])
            target.append(values[enc_end:dec_end])

        def _tensor(data: List[np.ndarray]) -> torch.Tensor:
            return torch.as_tensor(np.asarray(data), dtype=torch.float32)

        return (
            _tensor(enc_x).unsqueeze(-1),
            _tensor(dec_x).unsqueeze(-1),
            _tensor(enc_m).long(),
            _tensor(dec_m).long(),
            _tensor(target),
        )

    def _build_training_set(
        self,
        values: np.ndarray,
        marks: np.ndarray,
    ) -> Tuple[TensorDataset, TensorDataset]:
        n_val_windows = max(1, int(self.val_ratio * (len(values) - self.total_input_len + 1)))
        last_origin = len(values) - self.total_input_len
        val_origins = np.arange(max(0, last_origin - n_val_windows), last_origin + 1)
        train_origins = np.arange(0, max(0, val_origins[0]))

        if len(train_origins) == 0:  # series too short for a holdout
            train_origins = val_origins

        def _make(origins: np.ndarray) -> TensorDataset:
            enc_x, dec_x, enc_m, dec_m, target = self._windows(values, marks, origins)
            return TensorDataset(enc_x, enc_m, dec_x, dec_m, target)

        return _make(train_origins), _make(val_origins)

    # -- public API ------------------------------------------------------
    def fit(
        self,
        values: Union[np.ndarray, List[float], pd.Series],
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> "InformerForecaster":
        """Treinar a rede em uma série univariada, com early stopping e decaimento da taxa de aprendizado.
        """
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(values) < self.total_input_len + 2:
            raise ValueError(
                f"series with {len(values)} points is too short for seq_len="
                f"{self.seq_len} and pred_len={self.pred_len}"
            )

        marks = time_features(timestamps, len(values))
        self.mean_ = float(np.mean(values))
        self.std_ = float(np.std(values)) or 1.0
        values_std = self._standardize(values).astype(np.float32)

        train_ds, val_ds = self._build_training_set(values_std, marks)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.model = Informer(
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.pred_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            d_ff=self.d_ff,
            factor=self.factor,
            dropout=self.dropout,
            distil=self.distil,
            n_encoder_stacks=self.n_encoder_stacks,
        ).to(self.device)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # The paper decays the learning rate by 0.5 every epoch.
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        best_val = np.inf
        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        epochs_without_improvement = 0

        start = time.time()
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_losses = []
            for batch in train_loader:
                batch = [tensor.to(self.device) for tensor in batch]
                optimizer.zero_grad()
                output = self.model(*batch[:4])
                loss = F.mse_loss(output.squeeze(-1), batch[4])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            val_loss = self.evaluate(val_loader)
            scheduler.step()

            self.history["train_loss"].append(float(np.mean(train_losses)))
            self.history["val_loss"].append(float(val_loss))

            if self.verbose:
                print(
                    f"epoch {epoch:02d} | train mse {np.mean(train_losses):.5f} "
                    f"| val mse {val_loss:.5f} | lr {scheduler.get_last_lr()[0]:.2e}"
                )

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    if self.verbose:
                        print(f"early stopping at epoch {epoch} (best val mse {best_val:.5f})")
                    break

        self.model.load_state_dict(best_state)
        self.time_exec["training"] = time.time() - start
        return self

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        """Compute the MSE of the network over a data loader."""
        if self.model is None:
            raise RuntimeError("model is not fitted")
        self.model.eval()
        losses = []
        for batch in loader:
            batch = [tensor.to(self.device) for tensor in batch]
            output = self.model(*batch[:4])
            losses.append(F.mse_loss(output.squeeze(-1), batch[4]).item())
        return float(np.mean(losses)) if losses else np.inf

    @torch.no_grad()
    def predict(
        self,
        context_values: Union[np.ndarray, List[float], pd.Series],
        context_timestamps: Optional[pd.DatetimeIndex] = None,
        pred_len: Optional[int] = None,
    ) -> np.ndarray:

        if self.model is None:
            raise RuntimeError("model is not fitted")

        # The network is trained to write exactly ``pred_len`` steps; longer horizons
        # would need a positional table and decoder layout it never saw.
        horizon = int(min(pred_len or self.pred_len, self.pred_len))
        context = np.asarray(context_values, dtype=np.float32).reshape(-1)[-self.seq_len :]
        if len(context) < self.seq_len:
            raise ValueError(f"context must have at least {self.seq_len} points")

        marks = time_features(context_timestamps, self.seq_len + horizon)
        enc_marks = torch.as_tensor(marks[: self.seq_len], dtype=torch.long, device=self.device).unsqueeze(0)

        dec_values = np.zeros(self.label_len + horizon, dtype=np.float32)
        dec_values[: self.label_len] = self._standardize(context)[-self.label_len :]
        dec_x = torch.as_tensor(dec_values, dtype=torch.float32, device=self.device).view(1, -1, 1)
        dec_marks = torch.as_tensor(
            marks[self.seq_len - self.label_len : self.seq_len + horizon], dtype=torch.long, device=self.device
        ).unsqueeze(0)

        enc_x = torch.as_tensor(self._standardize(context), dtype=torch.float32, device=self.device).view(1, -1, 1)

        self.model.eval()
        output = self.model(enc_x, enc_marks, dec_x, dec_marks)
        return self._destandardize(output.squeeze().cpu().numpy())[:horizon]

    def rolling_forecast(
        self,
        history_values: Union[np.ndarray, List[float], pd.Series],
        test_values: Union[np.ndarray, List[float], pd.Series],
        timestamps: Optional[pd.DatetimeIndex] = None,
        stride: Optional[int] = None,
    ) -> np.ndarray:

        if self.model is None:
            raise RuntimeError("model is not fitted")

        history = np.asarray(history_values, dtype=np.float32).reshape(-1)
        test = np.asarray(test_values, dtype=np.float32).reshape(-1)
        stride = int(stride or self.pred_len)
        full = np.concatenate([history, test])
        index = pd.DatetimeIndex(timestamps) if timestamps is not None else None
        if index is not None and len(index) != len(full):
            raise ValueError(
                f"timestamps must cover history + test ({len(full)} points), got {len(index)}"
            )

        predictions = np.empty(0, dtype=np.float32)
        start_test = time.time()
        for origin in range(len(history), len(full), stride):
            remaining = len(full) - origin
            horizon = min(self.pred_len, remaining)
            context = full[origin - self.seq_len : origin]
            context_timestamps = index[origin - self.seq_len : origin] if index is not None else None
            block = self.predict(context, context_timestamps, pred_len=horizon)
            predictions = np.concatenate([predictions, block])

        self.time_exec["testing"] = time.time() - start_test
        return predictions[: len(test)]


# ---------------------------------------------------------------------------
# Multivariate wrapper used by the paper-protocol ETT experiment
# ---------------------------------------------------------------------------


class _WindowDataset(Dataset):

    def __init__(
        self,
        values: np.ndarray,
        marks: np.ndarray,
        origins: np.ndarray,
        seq_len: int,
        label_len: int,
        pred_len: int,
    ) -> None:

        self.values = values
        self.marks = marks
        self.origins = np.asarray(origins, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        """Return one ``(enc_x, enc_marks, dec_x, dec_marks, target)`` window."""
        origin = int(self.origins[index])
        enc_end = origin + self.seq_len
        dec_end = enc_end + self.pred_len

        enc_x = self.values[origin:enc_end]
        dec_x = np.zeros(
            (self.label_len + self.pred_len, self.values.shape[1]), dtype=np.float32
        )
        dec_x[: self.label_len] = self.values[enc_end - self.label_len : enc_end]
        target = self.values[enc_end:dec_end]

        return (
            torch.from_numpy(enc_x),
            torch.from_numpy(self.marks[origin:enc_end].astype(np.int64)),
            torch.from_numpy(dec_x),
            torch.from_numpy(self.marks[enc_end - self.label_len : dec_end].astype(np.int64)),
            torch.from_numpy(target),
        )


class InformerMultivariateForecaster:

    def __init__(
        self,
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 24,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_layers: int = 2,
        d_ff: int = 2048,
        factor: int = 5,
        dropout: float = 0.1,
        distil: bool = True,
        n_encoder_stacks: int = 2,
        replica_layers: Optional[int] = 1,
        replica_input_fraction: float = 0.25,
        epochs: int = 8,
        batch_size: int = 32,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        patience: int = 3,
        seed: int = 42,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.e_layers = int(e_layers)
        self.d_layers = int(d_layers)
        self.d_ff = int(d_ff)
        self.factor = int(factor)
        self.dropout = float(dropout)
        self.distil = bool(distil)
        self.n_encoder_stacks = int(n_encoder_stacks)
        self.replica_layers = replica_layers
        self.replica_input_fraction = float(replica_input_fraction)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.verbose = bool(verbose)

        self.model: Optional[Informer] = None
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.time_exec: Dict[str, Optional[float]] = {"training": None, "testing": None}
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None

    # -- helpers ---------------------------------------------------------
    @property
    def total_input_len(self) -> int:
        """Minimum number of points required before the forecast window."""
        return self.seq_len + self.pred_len

    @staticmethod
    def _to_array(values: Union[np.ndarray, pd.DataFrame, List[List[float]]], name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"{name} must be 2-D (n_points, n_features), got shape {array.shape}")
        return array

    def _check_fitted(self) -> None:
        if self.model is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("forecaster is not fitted")

    def _standardize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_) / self.std_

    def _destandardize(self, values: np.ndarray) -> np.ndarray:
        return values * self.std_ + self.mean_

    def build_model(self, n_features: Optional[int] = None) -> Informer:
        if n_features is not None:
            self.n_features_ = int(n_features)
        if self.n_features_ is None:
            raise RuntimeError("n_features_ is not set")
        self.model = Informer(
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.pred_len,
            c_in=self.n_features_,
            c_out=self.n_features_,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            d_ff=self.d_ff,
            factor=self.factor,
            dropout=self.dropout,
            distil=self.distil,
            n_encoder_stacks=self.n_encoder_stacks,
            replica_layers=self.replica_layers,
            replica_input_fraction=self.replica_input_fraction,
        ).to(self.device)
        return self.model

    def _windows(
        self, values: np.ndarray, marks: np.ndarray, origins: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the tensors of a batch of window origins (vectorized)."""
        enc_x, dec_x, enc_m, dec_m, target = [], [], [], [], []
        n_features = values.shape[1]
        for origin in np.asarray(origins, dtype=np.int64):
            enc_end = int(origin) + self.seq_len
            dec_end = enc_end + self.pred_len

            enc_x.append(values[origin:enc_end])
            dec_values = np.zeros((self.label_len + self.pred_len, n_features), dtype=np.float32)
            dec_values[: self.label_len] = values[enc_end - self.label_len : enc_end]
            dec_x.append(dec_values)
            enc_m.append(marks[origin:enc_end])
            dec_m.append(marks[enc_end - self.label_len : dec_end])
            target.append(values[enc_end:dec_end])

        return (
            torch.as_tensor(np.stack(enc_x), dtype=torch.float32),
            torch.as_tensor(np.stack(enc_m), dtype=torch.long),
            torch.as_tensor(np.stack(dec_x), dtype=torch.float32),
            torch.as_tensor(np.stack(dec_m), dtype=torch.long),
            torch.as_tensor(np.stack(target), dtype=torch.float32),
        )

    # -- public API ------------------------------------------------------
    def fit(
        self,
        train_values: Union[np.ndarray, pd.DataFrame],
        train_timestamps: Optional[pd.DatetimeIndex] = None,
        val_values: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        val_timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> "InformerMultivariateForecaster":
        """Treinar a rede em uma série multivariada.

        """
        train = self._to_array(train_values, "train_values")
        if val_values is None:
            raise ValueError("val_values is required: the paper early-stops on a validation split")
        val = self._to_array(val_values, "val_values")
        if train.shape[1] != val.shape[1]:
            raise ValueError(f"train has {train.shape[1]} features, val has {val.shape[1]}")
        if len(train) < self.total_input_len + 1:
            raise ValueError(
                f"training series with {len(train)} points is too short for seq_len="
                f"{self.seq_len} and pred_len={self.pred_len}"
            )
        if len(val) < self.total_input_len:
            raise ValueError(
                f"validation series with {len(val)} points is too short for seq_len="
                f"{self.seq_len} and pred_len={self.pred_len}"
            )

        self.n_features_ = train.shape[1]
        self.mean_ = train.mean(axis=0).astype(np.float32)
        std = train.std(axis=0).astype(np.float32)
        std[std == 0] = 1.0
        self.std_ = std

        train_std = self._standardize(train).astype(np.float32)
        val_std = self._standardize(val).astype(np.float32)
        train_marks = time_features(train_timestamps, len(train))
        val_marks = time_features(val_timestamps, len(val))

        train_origins = np.arange(0, len(train) - self.total_input_len + 1)
        val_origins = np.arange(0, len(val) - self.total_input_len + 1)

        train_loader = DataLoader(
            _WindowDataset(train_std, train_marks, train_origins, self.seq_len, self.label_len, self.pred_len),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            _WindowDataset(val_std, val_marks, val_origins, self.seq_len, self.label_len, self.pred_len),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.build_model()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # Paper: the learning rate is halved every epoch.
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        best_val = np.inf
        best_state = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        epochs_without_improvement = 0

        start = time.time()
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_losses = []
            for batch in train_loader:
                batch = [tensor.to(self.device) for tensor in batch]
                optimizer.zero_grad()
                output = self.model(*batch[:4])
                loss = F.mse_loss(output, batch[4])
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_loss = self.evaluate(val_loader)
            scheduler.step()

            self.history["train_loss"].append(float(np.mean(train_losses)))
            self.history["val_loss"].append(float(val_loss))

            if self.verbose:
                print(
                    f"epoch {epoch:02d} | train mse {np.mean(train_losses):.5f} "
                    f"| val mse {val_loss:.5f} | lr {scheduler.get_last_lr()[0]:.2e}"
                )

            if val_loss < best_val - 1e-7:
                best_val = val_loss
                best_state = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    if self.verbose:
                        print(f"early stopping at epoch {epoch} (best val mse {best_val:.5f})")
                    break

        self.model.load_state_dict(best_state)
        self.time_exec["training"] = time.time() - start
        return self

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        """Compute the MSE (all variables) of the network over a data loader."""
        if self.model is None:
            raise RuntimeError("model is not fitted")
        self.model.eval()
        losses = []
        for batch in loader:
            batch = [tensor.to(self.device) for tensor in batch]
            output = self.model(*batch[:4])
            losses.append(F.mse_loss(output, batch[4]).item())
        return float(np.mean(losses)) if losses else np.inf

    @torch.no_grad()
    def predict_windows(
        self,
        values: Union[np.ndarray, pd.DataFrame],
        timestamps: Optional[pd.DatetimeIndex],
        origins: Sequence[int],
        batch_size: int = 64,
    ) -> np.ndarray:
        self._check_fitted()
        values = self._to_array(values, "values")
        if values.shape[1] != self.n_features_:
            raise ValueError(f"expected {self.n_features_} features, got {values.shape[1]}")
        origins = np.asarray(list(origins), dtype=np.int64)
        marks = time_features(timestamps, len(values))
        values_std = self._standardize(values).astype(np.float32)

        # ProbSparse samples key/query pairs at every forward pass; resetting the
        # seed makes the whole evaluation reproducible.
        torch.manual_seed(self.seed)
        self.model.eval()
        chunks = []
        for start in range(0, len(origins), batch_size):
            enc_x, enc_m, dec_x, dec_m, _ = self._windows(
                values_std, marks, origins[start : start + batch_size]
            )
            output = self.model(
                enc_x.to(self.device),
                enc_m.to(self.device),
                dec_x.to(self.device),
                dec_m.to(self.device),
            )
            chunks.append(self._destandardize(output.cpu().numpy()))
        return np.concatenate(chunks, axis=0)

    @torch.no_grad()
    def predict(
        self,
        context_values: Union[np.ndarray, pd.DataFrame],
        context_timestamps: Optional[pd.DatetimeIndex] = None,
        pred_len: Optional[int] = None,
    ) -> np.ndarray:
        self._check_fitted()
        horizon = int(min(pred_len or self.pred_len, self.pred_len))
        context = self._to_array(context_values, "context_values")[-self.seq_len :]
        if len(context) < self.seq_len:
            raise ValueError(f"context must have at least {self.seq_len} points")

        marks = time_features(context_timestamps, self.seq_len + horizon)
        context_std = self._standardize(context)
        dec_values = np.zeros((self.label_len + horizon, self.n_features_), dtype=np.float32)
        dec_values[: self.label_len] = context_std[-self.label_len :]

        enc_x = torch.as_tensor(context_std, dtype=torch.float32).unsqueeze(0).to(self.device)
        dec_x = torch.as_tensor(dec_values, dtype=torch.float32).unsqueeze(0).to(self.device)
        enc_marks = torch.as_tensor(marks[: self.seq_len], dtype=torch.long).unsqueeze(0).to(self.device)
        dec_marks = torch.as_tensor(
            marks[self.seq_len - self.label_len : self.seq_len + horizon], dtype=torch.long
        ).unsqueeze(0).to(self.device)

        torch.manual_seed(self.seed)
        self.model.eval()
        output = self.model(enc_x, enc_marks, dec_x, dec_marks)
        return self._destandardize(output.squeeze(0).cpu().numpy())[:horizon]

    # -- persistence -----------------------------------------------------
    def _constructor_kwargs(self) -> Dict[str, object]:
        return {
            "seq_len": self.seq_len,
            "label_len": self.label_len,
            "pred_len": self.pred_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "e_layers": self.e_layers,
            "d_layers": self.d_layers,
            "d_ff": self.d_ff,
            "factor": self.factor,
            "dropout": self.dropout,
            "distil": self.distil,
            "n_encoder_stacks": self.n_encoder_stacks,
            "replica_layers": self.replica_layers,
            "replica_input_fraction": self.replica_input_fraction,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "seed": self.seed,
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save the weights, the standardization stats and the configuration."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self._constructor_kwargs(),
                "state_dict": self.model.state_dict(),
                "mean": self.mean_,
                "std": self.std_,
                "n_features": self.n_features_,
                "history": self.history,
                "time_exec": self.time_exec,
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls, path: Union[str, Path], device: str = "cpu", verbose: bool = False
    ) -> "InformerMultivariateForecaster":
        """Restore a forecaster saved with :meth:`save_checkpoint`."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        config = dict(payload["config"])
        config.update({"device": device, "verbose": verbose})
        forecaster = cls(**config)
        forecaster.mean_ = np.asarray(payload["mean"], dtype=np.float32)
        forecaster.std_ = np.asarray(payload["std"], dtype=np.float32)
        forecaster.n_features_ = int(payload["n_features"])
        forecaster.build_model()
        forecaster.model.load_state_dict(payload["state_dict"])
        forecaster.model.eval()
        forecaster.history = payload.get("history", forecaster.history)
        forecaster.time_exec = payload.get("time_exec", forecaster.time_exec)
        return forecaster
