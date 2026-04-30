import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import os


save_dir = "models/resnet18_CIFAR100"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# No normalization
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# CIFAR100 dataset
trainset = torchvision.datasets.CIFAR100(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

testset = torchvision.datasets.CIFAR100(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

model = models.resnet18()
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
# 100 output features for CIFAR100
model.fc = nn.Linear(512, 100)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[100, 150],
    gamma=0.1
)

epochs = 200
best_acc = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(trainloader, desc=f"Epoch [{epoch+1}/{epochs}]")

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_postfix({
            "Batch Loss": f"{loss.item():.4f}",
            "LR": optimizer.param_groups[0]['lr']
        })

    epoch_loss = running_loss / len(trainloader)
    print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {epoch_loss:.4f}")

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
    print(f"Epoch [{epoch+1}/{epochs}] Test Accuracy: {test_acc:.2f}%")

    if test_acc > best_acc:
        best_acc = test_acc
        # save the best model
        torch.save(model.state_dict(), os.path.join(save_dir, "resnet18_cifar100_best.pth"))
        print(f"Best model saved with accuracy: {best_acc:.2f}%")

    scheduler.step()

print("Model saved successfully.")
print(f"Best Test Accuracy: {best_acc:.2f}%")