import argparse
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models.quantization import resnet18 as qresnet18


class TinyImageNetValDataset(Dataset):
    """Standard Tiny ImageNet validation loader using val_annotations.txt."""

    def __init__(self, val_dir: str, class_to_idx: Dict[str, int], transform=None):
        self.val_dir = Path(val_dir)
        self.images_dir = self.val_dir / "images"
        self.transform = transform
        self.samples = []

        ann_file = self.val_dir / "val_annotations.txt"
        if not ann_file.exists():
            raise FileNotFoundError(f"Tiny ImageNet annotation file not found: {ann_file}")

        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_name, wnid = parts[0], parts[1]
                if wnid not in class_to_idx:
                    continue
                label = class_to_idx[wnid]
                self.samples.append((img_name, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_name, label = self.samples[idx]
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_pickle(path: str) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        feature_maps_dict = pickle.load(handle)
    if not isinstance(feature_maps_dict, dict) or len(feature_maps_dict) == 0:
        raise ValueError("Feature map pickle must contain a non-empty dictionary.")
    return feature_maps_dict


def feature_map_to_tensor(feature_map: Any) -> torch.Tensor:
    if isinstance(feature_map, torch.Tensor):
        tensor = feature_map.detach().cpu().float()
    elif isinstance(feature_map, np.ndarray):
        arr = feature_map
        if arr.ndim == 2:
            tensor = torch.from_numpy(arr).unsqueeze(0).float()
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3):
                tensor = torch.from_numpy(arr).float()
            elif arr.shape[-1] in (1, 3):
                tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
            else:
                raise ValueError(f"Unsupported numpy feature map shape: {arr.shape}")
        else:
            raise ValueError(f"Unsupported numpy feature map rank: {arr.ndim}")
    else:
        tensor = transforms.ToTensor()(feature_map).float()

    if tensor.ndim != 3:
        raise ValueError(f"Feature map tensor must be 3D (C,H,W), got shape {tuple(tensor.shape)}")

    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)

    if tensor.shape[0] != 3:
        raise ValueError(f"Feature map tensor must have 1 or 3 channels, got {tensor.shape[0]}")

    return tensor


def sample_residual(feature_maps_dict: Dict[str, Any]) -> torch.Tensor:
    feature_map_name = random.choice(list(feature_maps_dict.keys()))
    feature_map = feature_maps_dict[feature_map_name]
    return feature_map_to_tensor(feature_map)


def build_transforms(dataset: str):
    if dataset == "Tiny":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    if dataset == "CIFAR10":
        return transforms.Compose([
            transforms.ToTensor(),
        ])

    raise ValueError(f"Unsupported dataset: {dataset}")


def build_dataloader(dataset: str, data_root_path: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, int]:
    transform = build_transforms(dataset)

    if dataset == "CIFAR10":
        testset = torchvision.datasets.CIFAR10(
            root=data_root_path,
            train=False,
            download=True,
            transform=transform,
        )
        test_loader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return test_loader, 10

    if dataset == "Tiny":
        data_root = Path(data_root_path) / "tiny-imagenet-200"
        train_dir = data_root / "train"
        val_dir = data_root / "val"

        trainset = torchvision.datasets.ImageFolder(root=str(train_dir))
        val_set = TinyImageNetValDataset(
            val_dir=str(val_dir),
            class_to_idx=trainset.class_to_idx,
            transform=transform,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
        return val_loader, 200

    raise ValueError(f"Unsupported dataset: {dataset}")


def build_quantized_resnet18(dataset: str, num_classes: int, backend: str) -> nn.Module:
    # torch.backends.quantized.engine = backend

    model = qresnet18(weights=None, quantize=False)

    if dataset == "CIFAR10":
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.eval()

    try:
        model.fuse_model(is_qat=False)
    except TypeError:
        model.fuse_model()

    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    model.eval()
    return model


def load_int8_model(dataset: str, model_path: str, num_classes: int, backend: str) -> nn.Module:
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        model.eval()
        return model.cpu()

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unsupported checkpoint format. Expected a quantized model state_dict or a serialized nn.Module."
        )

    model = build_quantized_resnet18(dataset=dataset, num_classes=num_classes, backend=backend)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model.cpu()


@torch.no_grad()
def make_adv_batch(clean_batch: torch.Tensor, feature_maps_dict: Dict[str, Any], impact: float) -> torch.Tensor:
    _, c, h, w = clean_batch.shape
    adv_images = []

    for image in clean_batch:
        gb_tensor = sample_residual(feature_maps_dict)

        gb_h, gb_w = gb_tensor.shape[1], gb_tensor.shape[2]
        if gb_h != h or gb_w != w:
            gb_tensor = gb_tensor.unsqueeze(0)
            gb_tensor = F.interpolate(gb_tensor, size=(h, w), mode="bilinear", align_corners=False)
            gb_tensor = gb_tensor.squeeze(0)

        adv_image = torch.add(image, gb_tensor, alpha=impact)

        denom = torch.max(torch.abs(adv_image)).clamp_min(1e-12)
        adv_image = adv_image / denom

        if adv_image.shape[0] != c:
            raise RuntimeError(
                f"Adversarial image channel mismatch: expected {c}, got {adv_image.shape[0]}"
            )

        adv_images.append(adv_image)

    return torch.stack(adv_images, dim=0).float()


@torch.no_grad()
def evaluate_attack(
    model: nn.Module,
    data_loader: DataLoader,
    feature_maps_dict: Dict[str, Any],
    impact_of_residual: float,
    limit_batches: int = -1,
):
    clean_correct = 0
    total_samples = 0
    attacked_samples = 0
    still_correct_after_attack = 0

    for batch_idx, (inputs, labels) in enumerate(tqdm(data_loader, desc="Evaluating attack")):
        if limit_batches > 0 and batch_idx >= limit_batches:
            break

        inputs = inputs.cpu()
        labels = labels.cpu()

        clean_outputs = model(inputs)
        clean_pred = clean_outputs.argmax(dim=1)

        clean_correct += (clean_pred == labels).sum().item()
        total_samples += labels.size(0)

        idx_to_attack = (clean_pred == labels).nonzero(as_tuple=False).flatten()
        if idx_to_attack.numel() == 0:
            continue

        clean_subset = inputs[idx_to_attack]
        target_subset = clean_pred[idx_to_attack]

        adv_batch = make_adv_batch(clean_subset, feature_maps_dict, impact=impact_of_residual)
        adv_outputs = model(adv_batch)
        adv_pred = adv_outputs.argmax(dim=1)

        # ====================================================
        # 新增：肉眼测试（Eyeball Test）代码块
        # ====================================================
        # 只在第一个 batch 抓取第一张图片
        if batch_idx == 0 and clean_subset.size(0) > 0:
            import torchvision.utils as vutils

            # 定义反归一化，将图像还原回 0~1 的可视范围
            inv_normalize = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )

            # 取当前 batch 的第一张图
            orig_vis = inv_normalize(clean_subset[0])
            adv_vis = inv_normalize(adv_batch[0])

            # 强制截断到 0-1 之间，模拟真实的图像保存过程
            orig_vis = torch.clamp(orig_vis, 0, 1)
            adv_vis = torch.clamp(adv_vis, 0, 1)

            # 计算 L-infinity Norm (最大像素偏差)
            l_inf = torch.max(torch.abs(orig_vis - adv_vis)).item()
            print(f"\n\n[监控] 抓取成功！L-infinity 最大扰动: {l_inf:.4f} (标准隐蔽要求通常 < 0.031)")

            # 拼接并保存图像 (左：原图，右：对抗样本)
            comparison = torch.cat([orig_vis, adv_vis], dim=2)
            vutils.save_image(comparison, "adversarial_eyeball_test.png")
            print("[监控] 对比图已保存至 adversarial_eyeball_test.png，请立即查看！\n")
        # ====================================================

        attacked_samples += target_subset.size(0)
        still_correct_after_attack += (adv_pred == target_subset).sum().item()

    clean_acc = 100.0 * clean_correct / max(total_samples, 1)
    adv_acc_over_total = 100.0 * still_correct_after_attack / max(total_samples, 1)
    adv_acc_over_attacked = 100.0 * still_correct_after_attack / max(attacked_samples, 1)
    asr = 100.0 * (attacked_samples - still_correct_after_attack) / max(attacked_samples, 1)

    return {
        "clean_acc": clean_acc,
        "adv_acc_over_total": adv_acc_over_total,
        "adv_acc_over_attacked": adv_acc_over_attacked,
        "attack_success_rate": asr,
        "total_samples": total_samples,
        "attacked_samples": attacked_samples,
    }


def get_args():
    parser = argparse.ArgumentParser(description="ZQBA main_attack for INT8 ResNet18 on CIFAR10 or Tiny ImageNet")
    parser.add_argument("--feature-maps-path", type=str, required=True, help="Path to the pickle containing feature maps")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved INT8 checkpoint")
    parser.add_argument("--dataset", type=str, choices=["CIFAR10", "Tiny"], required=True, help="Target dataset")
    parser.add_argument("--data-root-path", type=str, default="./data", help="Root directory containing dataset files")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--impact-of-residual", type=float, default=0.4, help="Alpha used in clean + alpha * residual")
    parser.add_argument("--backend", type=str, default="fbgemm", choices=["fbgemm", "qnnpack"], help="Quantized backend")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for residual sampling")
    parser.add_argument("--limit-batches", type=int, default=-1, help="Optional debug limit on the number of batches")
    return parser.parse_args()


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    feature_maps_dict = load_pickle(args.feature_maps_path)
    data_loader, num_classes = build_dataloader(
        dataset=args.dataset,
        data_root_path=args.data_root_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = load_int8_model(
        dataset=args.dataset,
        model_path=args.model_path,
        num_classes=num_classes,
        backend=args.backend,
    )

    metrics = evaluate_attack(
        model=model,
        data_loader=data_loader,
        feature_maps_dict=feature_maps_dict,
        impact_of_residual=args.impact_of_residual,
        limit_batches=args.limit_batches,
    )

    print("=" * 60)
    print(f"Dataset                : {args.dataset}")
    print(f"INT8 checkpoint        : {args.model_path}")
    print(f"Feature maps           : {args.feature_maps_path}")
    print(f"Residual impact (alpha): {args.impact_of_residual}")
    print(f"Quant backend          : {args.backend}")
    print("-" * 60)
    print(f"Clean Accuracy (%)           : {metrics['clean_acc']:.2f}")
    print(f"Adv Accuracy / total (%)     : {metrics['adv_acc_over_total']:.2f}")
    print(f"Adv Accuracy / attacked (%)  : {metrics['adv_acc_over_attacked']:.2f}")
    print(f"Attack Success Rate (%)      : {metrics['attack_success_rate']:.2f}")
    print(f"Total samples                : {metrics['total_samples']}")
    print(f"Attacked clean samples       : {metrics['attacked_samples']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
