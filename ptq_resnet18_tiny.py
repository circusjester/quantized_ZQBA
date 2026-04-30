import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.quantization import resnet18 as qresnet18
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
def evaluate(model, data_loader):
    """
    accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            # inferencing on CPU
            images, labels = images.cpu(), labels.cpu()

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
    # Path and configuration
    device = torch.device("cpu")
    data_root = "./data/tiny-imagenet-200"
    load_path = "models/resnet18_tiny/resnet18_tiny_best.pth"  # 训练好的 FP32 模型
    save_dir = "models/resnet18_tiny_PTQ"
    os.makedirs(save_dir, exist_ok=True)

    # data
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

    trainset = torchvision.datasets.ImageFolder(root=train_dir)

    val_set = TinyImageNetValDataset(
        val_dir=val_dir,
        class_to_idx=trainset.class_to_idx,
        transform=transform_val
    )

    # calibration DataLoader with small batch size
    calib_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=4)
    # validation DataLoader with large batch size
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # init. quantized model structure
    model = qresnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 200)

    # FP32 weight
    try:
        state_dict = torch.load(load_path, map_location=device)
        # strict=False 很重要，因为带有 QuantStub 的模型 state_dict keys 略有不同
        model.load_state_dict(state_dict, strict=False)
        print("成功加载预训练的 FP32 权重。")
    except Exception as e:
        print(f"权重加载失败: {e}")
        return

    model.eval()

    print("\n[1/5] 测试原始 FP32 模型...")
    fp32_accuracy = evaluate(model, val_loader)
    print(f"[*] FP32 模型在验证集上的准确率: {fp32_accuracy:.2f}%\n")

    # --- PTQ ---
    print("[2/5] 开始融合模型层...")
    model.fuse_model(is_qat=False)

    print("[3/5] 配置量化参数 (Observer)...")
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    print("[4/5] 开始校准 (Calibration)...")
    num_calibration_batches = 32  # 32 个 batch * 32 样本 = 1024 张图片
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(calib_loader, desc="Calibrating")):
            model(images)
            if i >= num_calibration_batches:
                break

    print("\n[5/5] 正在转换为 INT8 模型并评估...")
    int8_model = torch.quantization.convert(model, inplace=True)

    save_path = os.path.join(save_dir, "resnet18_tiny_int8_ptq.pth")
    torch.save(int8_model.state_dict(), save_path)
    print(f"INT8 模型已成功保存至: {save_path}\n")

    print("-> 正在测试 INT8 量化模型...")
    int8_accuracy = evaluate(int8_model, val_loader)

    print("\n" + "=" * 40)
    print(" 量化精度对比报告")
    print("=" * 40)
    print(f" FP32 Baseline Accuracy : {fp32_accuracy:.2f}%")
    print(f" INT8 PTQ Accuracy      : {int8_accuracy:.2f}%")
    print(f" Accuracy Drop          : {fp32_accuracy - int8_accuracy:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()