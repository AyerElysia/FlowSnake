"""Checkpoint-compatible Flow + HA-SMoE denoiser used by the mainline.

This module is the single supported denoiser implementation.  It preserves
the parameter names and computations of the signed source configuration:

* 256-dimensional point tokens;
* six self-attention -> cross-attention -> SwiGLU blocks;
* alternating global Perceiver and local sampled-feature context;
* diffusion-progress conditioning; and
* one contour-level route path for blocks 2/4/6;
* an always-on shared FFN plus E4 Top-2 residual experts in those blocks; and
* one dense residual displacement head.

There are no alternative MoE variants in this module.  In particular, the
mainline has no point/token routing, per-block routers or output-head mixture.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply adaptive normalization modulation."""

    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """RMS normalization with checkpoint-compatible parameter names."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    """Dense SwiGLU feed-forward layer used by every contour block."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.v = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.v(x)) * self.w1(x)))


class CyclicRoPE1D(nn.Module):
    """Rotary position embedding for a closed contour."""

    def __init__(self, head_dim: int, num_points: int = 128) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.num_points = int(num_points)
        freqs = 1.0 / (
            10000.0
            ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("freqs", freqs)

    def _build_cache(
        self,
        num_points: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = (
            torch.arange(num_points, device=device, dtype=torch.float32)
            * (2.0 * math.pi / num_points)
        )
        angles = torch.outer(positions, self.freqs.to(device))
        return angles.cos(), angles.sin()

    def apply_rotary(self, x: torch.Tensor) -> torch.Tensor:
        num_points = int(x.shape[2])
        cos_cache, sin_cache = self._build_cache(num_points, x.device)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos = cos_cache[:num_points].unsqueeze(0).unsqueeze(0)
        sin = sin_cache[:num_points].unsqueeze(0).unsqueeze(0)
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding shared by diffusion time and stage progress."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=t.device) * -scale
        )
        embedding = t.unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat(
            (torch.sin(embedding), torch.cos(embedding)),
            dim=-1,
        )
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1), "constant", 0)
        return embedding


class PerceiverCompressor(nn.Module):
    """Compress a feature map into 256 global context tokens."""

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 256,
        num_queries: int = 256,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, out_dim)
        self.queries = nn.Embedding(num_queries, out_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=8,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, image_feat: torch.Tensor) -> torch.Tensor:
        batch_size = int(image_feat.shape[0])
        image_tokens = image_feat.flatten(2).transpose(1, 2)
        image_tokens = self.input_proj(image_tokens)
        queries = self.queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        compressed, _ = self.cross_attn(
            query=queries,
            key=image_tokens,
            value=image_tokens,
        )
        return self.norm(compressed + queries)


class SeparatePointEmbedding(nn.Module):
    """Embed displacement coordinates and sampled image features separately."""

    def __init__(self, state_dim: int = 256, feature_dim: int = 256) -> None:
        super().__init__()
        coordinate_dim = state_dim // 4
        feature_embedding_dim = state_dim - coordinate_dim
        self.coord_embed = nn.Sequential(
            nn.Linear(2, coordinate_dim),
            nn.SiLU(),
            nn.Linear(coordinate_dim, coordinate_dim),
        )
        self.feat_embed = nn.Sequential(
            nn.Linear(feature_dim, feature_embedding_dim),
            nn.SiLU(),
            nn.Linear(feature_embedding_dim, feature_embedding_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        sampled_feat: torch.Tensor,
    ) -> torch.Tensor:
        coordinate_embedding = self.coord_embed(x_t)
        feature_embedding = self.feat_embed(sampled_feat.transpose(1, 2))
        return torch.cat((coordinate_embedding, feature_embedding), dim=-1)


class MainlineDiTBlock(nn.Module):
    """Self-attention -> image cross-attention -> shared/routed FFN."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_points: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.norm1 = RMSNorm(self.dim)
        self.norm2 = RMSNorm(self.dim)
        self.norm3 = RMSNorm(self.dim)
        self.qk_norm = RMSNorm(self.head_dim)
        self.rope = CyclicRoPE1D(self.head_dim, num_points=num_points)

        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.sa_out_proj = nn.Linear(self.dim, self.dim, bias=False)

        self.cross_q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.cross_k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.cross_v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.ca_out_proj = nn.Linear(self.dim, self.dim, bias=False)

        self.mlp = SwiGLU(dim=self.dim, dropout=dropout)
        # ``mlp`` is always on.  Blocks 2/4/6 attach contour-level residual
        # specialists here while preserving dense-checkpoint parameter names.
        self.routed_moe: nn.Module | None = None
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, 9 * self.dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        self._cached_k: Optional[torch.Tensor] = None
        self._cached_v: Optional[torch.Tensor] = None

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, dim = x.shape
        q = self.q_proj(x).view(
            batch_size, num_points, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch_size, num_points, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size, num_points, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = self.rope.apply_rotary(self.qk_norm(q))
        k = self.rope.apply_rotary(self.qk_norm(k))
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        output = (attention @ v).transpose(1, 2).contiguous().view(
            batch_size, num_points, dim
        )
        return self.sa_out_proj(output)

    def _cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_points, dim = x.shape
        q = self.cross_q_proj(x).view(
            batch_size, num_points, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = self.qk_norm(q)
        if self._cached_k is not None and self._cached_v is not None:
            k, v = self._cached_k, self._cached_v
        else:
            context_length = int(context.shape[1])
            k = self.cross_k_proj(context).view(
                batch_size, context_length, self.num_heads, self.head_dim
            ).transpose(1, 2)
            v = self.cross_v_proj(context).view(
                batch_size, context_length, self.num_heads, self.head_dim
            ).transpose(1, 2)
            k = self.qk_norm(k)
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        output = (attention @ v).transpose(1, 2).contiguous().view(
            batch_size, num_points, dim
        )
        return self.ca_out_proj(output)

    def set_kv_cache(self, context: torch.Tensor) -> None:
        batch_size, context_length, _ = context.shape
        k = self.cross_k_proj(context).view(
            batch_size, context_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.cross_v_proj(context).view(
            batch_size, context_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        self._cached_k = self.qk_norm(k)
        self._cached_v = v

    def clear_kv_cache(self) -> None:
        self._cached_k = None
        self._cached_v = None

    def forward(
        self,
        x: torch.Tensor,
        image_context: torch.Tensor,
        condition_embedding: torch.Tensor,
        route_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        modulation = self.adaLN_modulation(condition_embedding)
        (
            shift_sa,
            scale_sa,
            gate_sa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_ff,
            scale_ff,
            gate_ff,
        ) = modulation.chunk(9, dim=1)

        self_attention_input = modulate(self.norm1(x), shift_sa, scale_sa)
        x = x + gate_sa.unsqueeze(1) * self._self_attention(self_attention_input)
        cross_attention_input = modulate(self.norm2(x), shift_ca, scale_ca)
        x = x + gate_ca.unsqueeze(1) * self._cross_attention(
            cross_attention_input,
            image_context,
        )
        ffn_input = modulate(self.norm3(x), shift_ff, scale_ff)
        ffn_output = self.mlp(ffn_input)
        if self.routed_moe is not None:
            if route_logits is None:
                raise RuntimeError("HA-SMoE route logits are required")
            ffn_output = ffn_output + self.routed_moe(ffn_input, route_logits)
        elif route_logits is not None:
            raise RuntimeError("route logits supplied to a dense DiT block")
        return x + gate_ff.unsqueeze(1) * ffn_output

    def enable_ha_smoe(self) -> None:
        if self.routed_moe is not None:
            raise RuntimeError("this DiT block already has routed experts")
        from .ha_smoe import ContourSharedResidualExperts

        self.routed_moe = ContourSharedResidualExperts(
            dim=self.dim,
            num_experts=4,
            top_k=2,
            expert_hidden_dim=256,
            load_balance_weight=1e-2,
            specialization_weight=5e-3,
            expert_init_std=1e-4,
        )

    def reg_loss(self) -> torch.Tensor:
        if self.routed_moe is None:
            return self.adaLN_modulation[-1].weight.new_zeros(())
        return self.routed_moe.reg_loss()


class DenseResidualFinalHead(nn.Module):
    """Dense output layer used by the released mainline checkpoint."""

    def __init__(
        self,
        dim: int = 256,
        out_dim: int = 2,
        hidden_dim: int = 1024,
        residual_init_std: float = 1e-4,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )
        self.residual_mlp = nn.Sequential(
            nn.Linear(dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), out_dim),
        )
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.xavier_uniform_(self.residual_mlp[0].weight)
        nn.init.zeros_(self.residual_mlp[0].bias)
        nn.init.normal_(self.residual_mlp[-1].weight, std=float(residual_init_std))
        nn.init.normal_(self.residual_mlp[-1].bias, std=float(residual_init_std))

    def forward(
        self,
        x: torch.Tensor,
        condition_embedding: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.adaLN(condition_embedding).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x) + self.residual_mlp(x)

    def reg_loss(self) -> torch.Tensor:
        return self.linear.weight.new_zeros(())


class MainlineFlowDenoiser(nn.Module):
    """Six-layer Flow denoiser with the single released HA-SMoE design."""

    def __init__(
        self,
        state_dim: int = 256,
        feature_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_points: int = 128,
        num_queries: int = 256,
        dense_residual_hidden_dim: int = 1024,
        use_s_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.feature_dim = int(feature_dim)
        self.time_dim = self.state_dim
        self.num_points = int(num_points)
        if int(num_layers) != 6:
            raise ValueError("HA-SMoE mainline requires exactly six DiT blocks")

        self.time_emb_net = nn.Sequential(
            SinusoidalTimeEmbedding(dim=self.state_dim // 4),
            nn.Linear(self.state_dim // 4, self.state_dim),
            nn.SiLU(),
            nn.Linear(self.state_dim, self.state_dim),
        )
        self.global_compressor = PerceiverCompressor(
            in_dim=self.feature_dim,
            out_dim=self.state_dim,
            num_queries=num_queries,
        )
        self.local_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.state_dim),
            nn.SiLU(),
            nn.Linear(self.state_dim, self.state_dim),
        )
        self.point_embed = SeparatePointEmbedding(
            state_dim=self.state_dim,
            feature_dim=self.feature_dim,
        )
        self.dit_layers = nn.ModuleList(
            [
                MainlineDiTBlock(
                    dim=self.state_dim,
                    num_heads=num_heads,
                    num_points=self.num_points,
                    dropout=0.0,
                )
                for _ in range(num_layers)
            ]
        )
        from .ha_smoe import ContourRoutePath

        # Human block numbers 2/4/6 map to zero-based Python indices 1/3/5.
        self._ha_smoe_layer_indices = (1, 3, 5)
        self._contour_moe_router = ContourRoutePath(
            dim=self.state_dim,
            num_routed_blocks=len(self._ha_smoe_layer_indices),
            num_experts=4,
            hidden_dim=256,
            temperature=1.0,
        )
        for index in self._ha_smoe_layer_indices:
            self.dit_layers[index].enable_ha_smoe()

        if not use_s_conditioning:
            raise ValueError("the mainline requires diffusion-stage conditioning")
        self.s_emb_net = nn.Sequential(
            SinusoidalTimeEmbedding(dim=self.state_dim // 4),
            nn.Linear(self.state_dim // 4, self.state_dim),
            nn.SiLU(),
            nn.Linear(self.state_dim, self.state_dim),
        )
        nn.init.zeros_(self.s_emb_net[-1].weight)
        nn.init.zeros_(self.s_emb_net[-1].bias)

        self.final_layer = DenseResidualFinalHead(
            dim=self.state_dim,
            out_dim=2,
            hidden_dim=dense_residual_hidden_dim,
            residual_init_std=1e-4,
        )

    def forward(
        self,
        cnn_feature: torch.Tensor,
        sampled_feat: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        adj=None,
        polys=None,
        py_ind: torch.Tensor | None = None,
        contour_scale: torch.Tensor | None = None,
        detail_feat: torch.Tensor | None = None,
        x_self_cond: torch.Tensor | None = None,
        locate_point_ctx: torch.Tensor | None = None,
        locate_global_ctx: torch.Tensor | None = None,
        locate_only: bool = False,
        s: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del adj, polys, contour_scale, detail_feat, x_self_cond
        if locate_point_ctx is not None or locate_global_ctx is not None or locate_only:
            raise ValueError("LocateToken conditioning is not part of the mainline")
        if x_t.ndim != 3 or x_t.shape[-1] != 2:
            raise ValueError(f"expected x_t [N,P,2], got {tuple(x_t.shape)}")
        if t.ndim != 1:
            raise ValueError(f"expected t [N], got {tuple(t.shape)}")
        if sampled_feat.ndim != 3:
            raise ValueError(
                f"expected sampled_feat [N,C,P], got {tuple(sampled_feat.shape)}"
            )
        if s is None:
            raise ValueError("the mainline denoiser requires stage progress s")

        parameter_dtype = next(self.parameters()).dtype
        cnn_feature = cnn_feature.to(parameter_dtype)
        sampled_feat = sampled_feat.to(parameter_dtype)
        x_t = x_t.to(parameter_dtype)
        t = t.to(parameter_dtype)
        s = s.to(parameter_dtype)

        contour_count = int(x_t.shape[0])
        time_embedding = self.time_emb_net(t)
        stage_embedding = self.s_emb_net(
            (s * 1000.0).to(device=t.device, dtype=time_embedding.dtype)
        )
        condition_embedding = time_embedding + stage_embedding

        if cnn_feature.ndim == 3:
            cnn_feature = cnn_feature.unsqueeze(0)
        global_context = self.global_compressor(cnn_feature)
        if py_ind is not None:
            global_context = global_context[py_ind]
        elif global_context.shape[0] != contour_count:
            if global_context.shape[0] == 1:
                global_context = global_context.expand(contour_count, -1, -1)
            else:
                raise ValueError(
                    "image/contour batch mismatch: "
                    f"global={global_context.shape[0]}, contours={contour_count}"
                )

        local_context = self.local_proj(sampled_feat.transpose(1, 2))
        x = self.point_embed(x_t, sampled_feat)
        route_path = self._contour_moe_router(local_context)
        route_positions = {
            layer_index: route_index
            for route_index, layer_index in enumerate(self._ha_smoe_layer_indices)
        }
        for index, layer in enumerate(self.dit_layers):
            context = global_context if index % 2 == 0 else local_context
            route_logits = (
                route_path[:, route_positions[index], :]
                if index in route_positions
                else None
            )
            x = layer(
                x,
                context,
                condition_embedding,
                route_logits=route_logits,
            )

        prediction = self.final_layer(x, condition_embedding)
        regularization = self.final_layer.reg_loss()
        for layer in self.dit_layers:
            regularization = regularization + layer.reg_loss()
        return prediction, regularization

    def moe_diagnostics(self) -> dict[str, torch.Tensor]:
        from .ha_smoe import collect_ha_smoe_diagnostics

        return collect_ha_smoe_diagnostics(self)


__all__ = (
    "DenseResidualFinalHead",
    "MainlineDiTBlock",
    "MainlineFlowDenoiser",
)
