"""Data utilities: datasets, transforms, and image encoding helpers."""

import base64
import io
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


# CIFAR-10 normalization constants
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms(train: bool) -> transforms.Compose:
    """Builds data transforms for CIFAR-10 sized images.

    - Train: random crop with padding, random horizontal flip, ToTensor, Normalize
    - Eval:  resize to 32x32 (safety), ToTensor, Normalize

    TODO: Consider RandAugment, CutMix/MixUp for more accuracy if needed.
    """

    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
            ]
        )


@dataclass
class DatasetInfo:
    classes: List[str]
    num_classes: int


def build_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DatasetInfo]:
    """Builds CIFAR-10 train/val dataloaders and returns dataset info."""

    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=build_transforms(train=True),
    )
    val_ds = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=build_transforms(train=False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    dataset_info = DatasetInfo(classes=train_ds.classes, num_classes=len(train_ds.classes))
    return train_loader, val_loader, dataset_info


def pil_to_jpeg_base64(image: Image.Image, quality: int = 90) -> Tuple[str, int, int]:
    """Encodes a PIL image as base64 JPEG string with given quality.

    Returns (base64_string, width, height)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    width, height = image.size
    return encoded, width, height


def base64_to_pil(image_b64: str) -> Image.Image:
    """Decodes a base64 image (JPEG/PNG) into a PIL image."""
    raw = base64.b64decode(image_b64)
    buffer = io.BytesIO(raw)
    image = Image.open(buffer)
    return image


def prepare_tensor_for_model(image: Image.Image) -> torch.Tensor:
    """Preprocesses PIL image to model-ready tensor (normalized, shape [1, C, H, W])."""
    eval_tfms = build_transforms(train=False)
    tensor = eval_tfms(image).unsqueeze(0)  # add batch dim
    return tensor

