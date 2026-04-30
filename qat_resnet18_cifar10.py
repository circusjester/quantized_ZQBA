import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.quantization import resnet18 as qresnet18
import os
from tqdm import tqdm

# Configuration and path
device = torch.device("cpu") # 注意：PyTorch 官方量化逻辑主要在 CPU 上准备和转换
save_dir = "models/resnet18_CIFAR10_QAT"
os.makedirs(save_dir, exist_ok=True)
load_path = "models/resnet18_CIFAR10/resnet18_cifar10_best.pth" # 指向您之前训练好的模型

# Data process
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Pretrained quantized resnet18
# qresnet18 with QuantStub/DeQuantStub and FloatFunctional
model = qresnet18(weights=None, num_classes=10)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

try:
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("Pre-trained FP32 weights loaded.")
except:
    print("Warning: Direct load failed, ensure weight keys match.")

# QAT preparation
model.train()
# Conv+BN+ReLU
model.fuse_model()
# fbgemm for x86
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
# 准备 QAT (插入伪量化节点)
torch.quantization.prepare_qat(model, inplace=True)

# QAT optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

epochs_qat = 10
for epoch in range(epochs_qat):
    model.train()
    # 在最后几个 epoch 冻结量化参数以稳定收敛
    if epoch > 8:
        model.apply(torch.quantization.disable_observer)
    if epoch > 9:
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    for images, labels in tqdm(trainloader, desc=f"QAT Epoch {epoch+1}"):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# # save the fake quantized model before conversion, for the use of guided backprop (if used)
# torch.save(model.state_dict(), os.path.join(save_dir, "resnet18_qat_before_convert.pth"))
# print("QAT (float, fake quant) model saved.")


# convert to int8
model.eval()
model.cpu()
int8_model = torch.quantization.convert(model, inplace=False)

torch.save(int8_model.state_dict(), os.path.join(save_dir, "resnet18_int8_final.pth"))
print("INT8 model saved. Ready for ZQBA attack analysis.")