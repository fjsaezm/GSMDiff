"""High-pass filter overlays matching the filters used by the legacy GSM loss."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional


def _as_channel_image(image: Tensor) -> Tensor:
    if image.ndim == 2:
        return image.unsqueeze(0)
    if image.ndim != 3:
        raise ValueError(
            "image must have shape (height, width) or (channels, height, width); "
            f"got {tuple(image.shape)}."
        )
    return image


def high_pass_magnitude(image: Tensor) -> Tensor:
    """Return the combined horizontal/vertical first-difference response.

    These are the same ``[1, -1]`` and ``[1, -1]^T`` filters used by
    ``GSMLoss``. Responses are combined across color channels with a root mean
    square, producing one ``(height, width)`` edge-strength image.
    """
    image = _as_channel_image(image)
    horizontal = image[..., :, :-1] - image[..., :, 1:]
    vertical = image[..., :-1, :] - image[..., 1:, :]
    horizontal = functional.pad(horizontal, (0, 1, 0, 0))
    vertical = functional.pad(vertical, (0, 0, 0, 1))
    return torch.sqrt((horizontal.square() + vertical.square()).mean(dim=0))


def high_pass_coefficients(image: Tensor) -> Tensor:
    """Return pooled signed coefficients from both first-difference filters."""
    image = _as_channel_image(image)
    horizontal = image[..., :, :-1] - image[..., :, 1:]
    vertical = image[..., :-1, :] - image[..., 1:, :]
    return torch.cat((horizontal.reshape(-1), vertical.reshape(-1)))


def draw_high_pass_contours(
    axis: Any,
    image: Tensor,
    *,
    quantile: float = 0.9,
    threshold: float | None = None,
    color: str = "#ffff00",
    opacity: float = 1.0,
) -> Tensor:
    """Draw thresholded high-pass responses over an image axis.

    The underlying image should already have been drawn on ``axis``. If an
    absolute ``threshold`` is not supplied, it is estimated independently for
    the image using the requested response quantile. The returned CPU boolean
    tensor is the exact mask painted by this function.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}.")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError(f"opacity must be in [0, 1], got {opacity}.")
    if threshold is not None and threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}.")

    response = high_pass_magnitude(image.detach()).float().cpu()
    cutoff = float(torch.quantile(response, quantile)) if threshold is None else threshold
    mask = response > cutoff

    from matplotlib.colors import to_rgb

    overlay = torch.zeros((*mask.shape, 4), dtype=torch.float32)
    overlay[..., :3] = torch.tensor(to_rgb(color), dtype=torch.float32)
    overlay[..., 3] = mask.to(torch.float32) * opacity
    axis.imshow(overlay.numpy(), interpolation="nearest")
    return mask
