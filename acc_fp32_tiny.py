import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader


# ==========================================
# Dataset
# ==========================================
class TinyImageNetValDataset(Dataset):
    """
    Tiny ImageNet validation
    """

    def __init__(self, val_dir, class_to_idx, transform=None):
        self.val_dir = Path(val_dir)
        self.images_dir = self.val_dir / "images"
        self.transform = transform
        self.samples = []

        ann_file = self.val_dir / "val_annotations.txt"
        with open(ann_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                img_name = parts[0]
                wnid = parts[1]
                if wnid in class_to_idx:
                    label = class_to_idx[wnid]
                    self.samples.append((img_name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ==========================================
# Evaluation
# ==========================================
def evaluate(model, data_loader, device):
    """
    accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating FP32 Model"):
            # CPU/GPU
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


# ==========================================
# Main Function
# ==========================================
def main():
    # --- path and configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = "./data/tiny-imagenet-200"
    load_path = "models/resnet18_tiny/resnet18_tiny_best.pth"  # 你的 FP32 模型权重路径

    # --- data process ---
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    print("Loading dataset mapping...")
    trainset = torchvision.datasets.ImageFolder(root=train_dir)

    # validation set
    val_set = TinyImageNetValDataset(
        val_dir=val_dir,
        class_to_idx=trainset.class_to_idx,
        transform=transform_val
    )

    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    print("Initializing ResNet18 model...")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 200)

    # FP32 weight
    if not os.path.exists(load_path):
        print(f"错误: 找不到权重文件 {load_path}")
        return

    try:
        state_dict = torch.load(load_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"成功加载预训练权重: {load_path}")
    except Exception as e:
        print(f"权重加载失败: {e}")
        return

    model.to(device)

    # --- evaluation starts ---
    print("\n开始测试模型在验证集上的准确率...")
    fp32_accuracy = evaluate(model, val_loader, device)

    print("\n" + "=" * 40)
    print(" 测试报告")
    print("=" * 40)
    print(f" Model       : ResNet18 (FP32)")
    print(f" Dataset     : Tiny ImageNet")
    print(f" Accuracy    : {fp32_accuracy:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()