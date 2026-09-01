"""Compute dataset-backed FID and KID over saved generated image batches."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _image_paths(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if path.is_dir():
        return sorted(
            item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES
        )
    return []


def _load_tensor(path: Path) -> Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        for key in ("samples", "images", "geometric_samples"):
            if isinstance(value.get(key), Tensor):
                value = value[key]
                break
    if not isinstance(value, Tensor):
        raise TypeError(f"{path} does not contain a tensor or recognized tensor mapping.")
    if value.ndim != 4:
        raise ValueError(f"{path} must contain NCHW images, got {tuple(value.shape)}.")
    return value


def _source_count(sources: Sequence[Path]) -> int:
    count = 0
    for source in sources:
        if source.suffix.lower() in {".pt", ".pth"}:
            count += len(_load_tensor(source))
        else:
            images = _image_paths(source)
            if not images:
                raise FileNotFoundError(f"No supported images found at {source}.")
            count += len(images)
    return count


def _tensor_to_uint8(images: Tensor, value_range: str) -> Tensor:
    images = images.detach().cpu()
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    if images.shape[1] != 3:
        raise ValueError(f"FID/KID require one or three channels, got {images.shape[1]}.")
    if value_range == "auto":
        if images.dtype == torch.uint8:
            value_range = "uint8"
        elif float(images.min()) < 0.0:
            value_range = "minus_one_one"
        elif float(images.max()) <= 1.0:
            value_range = "zero_one"
        else:
            raise ValueError("Cannot infer float image range; specify it explicitly.")
    if value_range == "uint8":
        if images.dtype != torch.uint8:
            raise ValueError("uint8 range requires a torch.uint8 tensor.")
        return images
    images = images.float()
    if value_range == "minus_one_one":
        images = (images.clamp(-1.0, 1.0) + 1.0) * 127.5
    elif value_range == "zero_one":
        images = images.clamp(0.0, 1.0) * 255.0
    else:
        raise ValueError(f"Unknown image range: {value_range}.")
    return images.round().to(torch.uint8)


def _load_image_batch(paths: Sequence[Path], image_size: int) -> Tensor:
    from PIL import Image

    images = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (image_size, image_size):
                image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
            images.append(torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1))
    return torch.stack(images)


def _batches(
    sources: Sequence[Path],
    *,
    batch_size: int,
    tensor_range: str,
    image_size: int,
) -> Iterator[Tensor]:
    for source in sources:
        if source.suffix.lower() in {".pt", ".pth"}:
            images = _load_tensor(source)
            for start in range(0, len(images), batch_size):
                yield _tensor_to_uint8(images[start : start + batch_size], tensor_range)
        else:
            paths = _image_paths(source)
            for start in range(0, len(paths), batch_size):
                yield _load_image_batch(paths[start : start + batch_size], image_size)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Inception-v3 FID and KID for generated and real image sources."
    )
    parser.add_argument("--real", type=Path, nargs="+", required=True)
    parser.add_argument("--fake", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-range", choices=("auto", "minus_one_one", "zero_one", "uint8"), default="auto")
    parser.add_argument("--fake-range", choices=("auto", "minus_one_one", "zero_one", "uint8"), default="minus_one_one")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--minimum-samples", type=int, default=1000)
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.image_size <= 0 or args.minimum_samples < 2:
        parser.error("batch size and image size must be positive; minimum samples must be >= 2")
    real_count = _source_count(args.real)
    fake_count = _source_count(args.fake)
    if min(real_count, fake_count) < args.minimum_samples:
        parser.error(
            f"found {real_count} real and {fake_count} fake images; both must contain at "
            f"least {args.minimum_samples}. Use more samples rather than lowering this guard "
            "for reported FID/KID."
        )
    if args.kid_subset_size > min(real_count, fake_count):
        parser.error("KID subset size cannot exceed the real or fake sample count")

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        from torchmetrics.wrappers import FeatureShare
    except ImportError as error:
        raise RuntimeError(
            'FID/KID dependencies are missing; install with pip install -e ".[metrics]".'
        ) from error

    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    fid = FrechetInceptionDistance(feature=2048, normalize=False).set_dtype(torch.float64)
    kid = KernelInceptionDistance(
        feature=2048,
        subsets=args.kid_subsets,
        subset_size=args.kid_subset_size,
        normalize=False,
    )
    metrics = FeatureShare({"fid": fid, "kid": kid}).to(device)
    for images in _batches(
        args.real,
        batch_size=args.batch_size,
        tensor_range=args.real_range,
        image_size=args.image_size,
    ):
        metrics.update(images.to(device), real=True)
    for images in _batches(
        args.fake,
        batch_size=args.batch_size,
        tensor_range=args.fake_range,
        image_size=args.image_size,
    ):
        metrics.update(images.to(device), real=False)
    values = metrics.compute()
    kid_mean, kid_std = values["kid"]
    result = {
        "fid": float(values["fid"].cpu()),
        "kid_mean": float(kid_mean.cpu()),
        "kid_std": float(kid_std.cpu()),
        "real_image_count": real_count,
        "fake_image_count": fake_count,
        "feature": 2048,
        "kid_subsets": args.kid_subsets,
        "kid_subset_size": args.kid_subset_size,
        "real_sources": [str(path) for path in args.real],
        "fake_sources": [str(path) for path in args.fake],
        "real_tensor_range": args.real_range,
        "fake_tensor_range": args.fake_range,
        "directory_image_size": args.image_size,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
