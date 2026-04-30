import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.quantization import resnet18 as qresnet18
from tqdm import tqdm


def test_int8_accuracy(model_path, data_root='./data'):
    device = torch.device("cpu")  # 注意：INT8 模型推理目前仅支持 CPU

    # 1. 准备数据加载器 (必须与训练时一致)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    # 2. 构建量化模型结构 (必须执行 fuse 和 prepare 流程，否则无法匹配 state_dict)
    print("正在构建 INT8 模型框架...")
    model = qresnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # 必须执行与训练一致的量化初始化步骤
    model.eval()
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    # 3. 关键：转换为 INT8 格式后再加载权重
    int8_model = torch.quantization.convert(model, inplace=False)

    # 4. 加载保存的 state_dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        int8_model.load_state_dict(state_dict)
        print(f"✓ 成功加载 INT8 模型: {model_path}")
    except Exception as e:
        print(f"× 加载失败: {e}")
        return

    # 5. 测试精确度
    int8_model.eval()
    correct = 0
    total = 0

    print("开始测试 INT8 模型精确度...")
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            # 推理
            outputs = int8_model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    print(f"\n" + "=" * 30)
    print(f"测试完成！")
    print(f"INT8 模型最终精确度: {acc:.2f}%")
    print(f"=" * 30)

    # 针对 ZQBA 研究的额外提示：
    # 如果你想查看某一层的整数权重，可以使用以下代码：
    # raw_int = int8_model.model.conv1.weight().int_repr()
    # print(f"第一层权重示例 (前5个): {raw_int.flatten()[:5]}")


if __name__ == "__main__":
    # 请确保路径指向你保存的 .pth 文件
    MODEL_PATH = "models/resnet18_CIFAR10_QAT/resnet18_int8_final.pth"
    test_int8_accuracy(MODEL_PATH)