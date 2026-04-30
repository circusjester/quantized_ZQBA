import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class TinyImageNetValDataset(Dataset):
    """
    Read the standard validation set of Tiny ImageNet
    data/tiny-imagenet-200/val/
        images/
        val_annotations.txt
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


def main():
    data_root = "./data/tiny-imagenet-200"
    save_dir = "models/resnet18_tiny"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Tiny ImageNet 常见做法：resize 到 224，再用 ImageNet 标准归一化
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    transform_test = transforms.Compose([
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

    # ImageFolder for training set
    trainset = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=transform_train
    )

    # val_annotations.txt for validation set
    valset = TinyImageNetValDataset(
        val_dir=val_dir,
        class_to_idx=trainset.class_to_idx,
        transform=transform_test
    )

    trainloader = DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    testloader = DataLoader(
        valset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 200 output features
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 200)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 80],
        gamma=0.1
    )

    epochs = 100
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(trainloader, desc=f"Epoch [{epoch + 1}/{epochs}]")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "LR": optimizer.param_groups[0]["lr"]
            })

        train_loss = running_loss / len(trainset)
        train_acc = 100.0 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100.0 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}] Val Accuracy: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "resnet18_tiny_best.pth"))
            print(f"Best model saved with accuracy: {best_acc:.2f}%")

        scheduler.step()

    print("Training finished.")
    print(f"Best Val Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()