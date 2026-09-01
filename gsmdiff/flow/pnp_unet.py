"""U-Net used by the public PnP-Flow optimal-transport checkpoints.

The module layout intentionally matches ``pnpflow.models.UNet`` so its published
state dictionaries load without key translation.  The architecture originates
from PnP-Flow (BSD-3-Clause), which in turn adapts sdeflow-light.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out


class Swish(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return torch.sigmoid(value) * value


def group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels, eps=1e-6, affine=True)


def _variance_scaling_(tensor: Tensor, scale: float) -> None:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_out  # PnP-Flow's ``fan_avg`` implementation resolves to fan_out.
    gain = 1e-10 if scale == 0 else scale
    bound = math.sqrt(3.0 * gain / max(1.0, fan))
    with torch.no_grad():
        tensor.uniform_(-bound, bound)


def _dense(in_channels: int, out_channels: int, init_scale: float = 1.0) -> nn.Linear:
    layer = nn.Linear(in_channels, out_channels)
    _variance_scaling_(layer.weight, init_scale)
    nn.init.zeros_(layer.bias)
    return layer


def _conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int] = (3, 3),
    stride: int = 1,
    padding: int = 1,
    init_scale: float = 1.0,
) -> nn.Conv2d:
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    _variance_scaling_(layer.weight, init_scale)
    nn.init.zeros_(layer.bias)
    return layer


def _sinusoidal_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape (batch,).")
    half_dim = embedding_dim // 2
    frequencies = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        * (-math.log(10000.0) / (half_dim - 1))
    )
    phases = timesteps.to(torch.get_default_dtype())[:, None] * frequencies[None, :]
    embedding = torch.cat((phases.sin(), phases.cos()), dim=1)
    if embedding_dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.main = nn.Sequential(
            _dense(embedding_dim, hidden_dim), Swish(), _dense(hidden_dim, output_dim)
        )

    def forward(self, timestep: Tensor) -> Tensor:
        return self.main(_sinusoidal_embedding(timestep, self.embedding_dim))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        temb_ch: int,
        out_ch: int | None = None,
        *,
        dropout: float = 0.0,
        normalize: Callable[[int], nn.Module] = group_norm,
    ) -> None:
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.act = Swish()
        self.temb_proj = _dense(temb_ch, out_ch)
        self.norm1 = normalize(in_ch)
        self.conv1 = _conv2d(in_ch, out_ch)
        self.norm2 = normalize(out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = _conv2d(out_ch, out_ch, init_scale=0.0)
        self.shortcut = (
            _conv2d(in_ch, out_ch, kernel_size=(1, 1), padding=0)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, value: Tensor, embedding: Tensor) -> Tensor:
        hidden = self.conv1(self.act(self.norm1(value)))
        hidden = hidden + self.temb_proj(self.act(embedding))[:, :, None, None]
        hidden = self.conv2(self.dropout(self.act(self.norm2(hidden))))
        return self.shortcut(value) + hidden


class SelfAttention(nn.Module):
    def __init__(self, channels: int, normalize: Callable[[int], nn.Module] = group_norm) -> None:
        super().__init__()
        self.norm = normalize(channels)
        self.attn_q = _conv2d(channels, channels, kernel_size=1, padding=0)
        self.attn_k = _conv2d(channels, channels, kernel_size=1, padding=0)
        self.attn_v = _conv2d(channels, channels, kernel_size=1, padding=0)
        self.proj_out = _conv2d(channels, channels, kernel_size=1, padding=0, init_scale=0.0)

    def forward(self, value: Tensor, _embedding: Tensor | None = None) -> Tensor:
        _, channels, height, width = value.shape
        hidden = self.norm(value)
        query = self.attn_q(hidden).view(-1, channels, height * width)
        key = self.attn_k(hidden).view(-1, channels, height * width)
        attention = torch.bmm(query.transpose(1, 2), key) * channels**-0.5
        attention = attention.softmax(dim=-1)
        hidden = torch.bmm(
            self.attn_v(hidden).view(-1, channels, height * width),
            attention.transpose(1, 2),
        ).view(-1, channels, height, width)
        return value + self.proj_out(hidden)


def _downsample(channels: int, with_conv: bool) -> nn.Module:
    return _conv2d(channels, channels, stride=2) if with_conv else nn.AvgPool2d(2, 2)


def _upsample(channels: int, with_conv: bool) -> nn.Module:
    modules = nn.Sequential()
    modules.add_module("up_nn", nn.Upsample(scale_factor=2, mode="nearest"))
    if with_conv:
        modules.add_module("up_conv", _conv2d(channels, channels))
    return modules


class PnPFlowUNet(nn.Module):
    """Checkpoint-compatible PnP-Flow velocity U-Net."""

    def __init__(
        self,
        input_channels: int = 3,
        input_height: int = 128,
        ch: int = 32,
        output_channels: int | None = None,
        ch_mult: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 6,
        attn_resolutions: Sequence[int] = (16, 8),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.ch = ch
        self.output_channels = input_channels if output_channels is None else output_channels
        self.ch_mult = tuple(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = tuple(attn_resolutions)
        self.temb_net = TimestepEmbedding(ch, ch * 4, ch * 4)
        self.begin_conv = _conv2d(input_channels, ch)

        resolution = input_height
        current_channels = ch
        skip_channels = [ch]
        down_modules: list[nn.ModuleDict] = []
        for level, multiplier in enumerate(self.ch_mult):
            blocks: dict[str, nn.Module] = {}
            output_ch = ch * multiplier
            for block in range(num_res_blocks):
                blocks[f"{level}a_{block}a_block"] = ResidualBlock(
                    current_channels, ch * 4, output_ch, dropout=dropout
                )
                if resolution in self.attn_resolutions:
                    blocks[f"{level}a_{block}b_attn"] = SelfAttention(output_ch)
                skip_channels.append(output_ch)
                current_channels = output_ch
            if level != len(self.ch_mult) - 1:
                blocks[f"{level}b_downsample"] = _downsample(output_ch, resamp_with_conv)
                resolution //= 2
                skip_channels.append(output_ch)
            down_modules.append(nn.ModuleDict(blocks))
        self.down_modules = nn.ModuleList(down_modules)

        self.mid_modules = nn.ModuleList(
            [
                ResidualBlock(current_channels, ch * 4, current_channels, dropout=dropout),
                SelfAttention(current_channels),
                ResidualBlock(current_channels, ch * 4, current_channels, dropout=dropout),
            ]
        )

        up_modules: list[nn.ModuleDict] = []
        for level in reversed(range(len(self.ch_mult))):
            blocks = {}
            output_ch = ch * self.ch_mult[level]
            for block in range(num_res_blocks + 1):
                blocks[f"{level}a_{block}a_block"] = ResidualBlock(
                    current_channels + skip_channels.pop(),
                    ch * 4,
                    output_ch,
                    dropout=dropout,
                )
                if resolution in self.attn_resolutions:
                    blocks[f"{level}a_{block}b_attn"] = SelfAttention(output_ch)
                current_channels = output_ch
            if level != 0:
                blocks[f"{level}b_upsample"] = _upsample(output_ch, resamp_with_conv)
                resolution *= 2
            up_modules.append(nn.ModuleDict(blocks))
        if skip_channels:
            raise RuntimeError("Internal U-Net skip-channel construction failed.")
        self.up_modules = nn.ModuleList(up_modules)
        self.end_conv = nn.Sequential(
            group_norm(current_channels),
            Swish(),
            _conv2d(current_channels, self.output_channels, init_scale=0.0),
        )

    def forward(self, value: Tensor, timestep: Tensor) -> Tensor:
        if timestep.ndim == 0:
            timestep = timestep.repeat(value.shape[0])
        embedding = self.temb_net(timestep)
        skips = [self.begin_conv(value)]
        for level, blocks in enumerate(self.down_modules):
            for block in range(self.num_res_blocks):
                hidden = blocks[f"{level}a_{block}a_block"](skips[-1], embedding)
                if hidden.shape[-1] in self.attn_resolutions:
                    hidden = blocks[f"{level}a_{block}b_attn"](hidden, embedding)
                skips.append(hidden)
            if level != len(self.ch_mult) - 1:
                skips.append(blocks[f"{level}b_downsample"](skips[-1]))

        hidden = skips[-1]
        for module in self.mid_modules:
            hidden = module(hidden, embedding)
        for index, level in enumerate(reversed(range(len(self.ch_mult)))):
            blocks = self.up_modules[index]
            for block in range(self.num_res_blocks + 1):
                hidden = blocks[f"{level}a_{block}a_block"](
                    torch.cat((hidden, skips.pop()), dim=1), embedding
                )
                if hidden.shape[-1] in self.attn_resolutions:
                    hidden = blocks[f"{level}a_{block}b_attn"](hidden, embedding)
            if level != 0:
                hidden = blocks[f"{level}b_upsample"](hidden)
        if skips:
            raise RuntimeError("Internal U-Net skip stack was not exhausted.")
        return self.end_conv(hidden)
