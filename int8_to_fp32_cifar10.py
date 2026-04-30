import torch
import torch.nn as nn
from torchvision.models.quantization import resnet18 as qresnet18


def build_int8_model(num_classes=10):
    model = qresnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def convert_int8_to_fp32(int8_model):
    fp32_model = int8_model.to(torch.float)

    # dequant
    for name, module in fp32_model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            try:
                if hasattr(module.weight, "dequantize"):
                    module.weight = torch.nn.Parameter(module.weight().dequantize())
            except:
                pass

    return fp32_model


def main():
    int8_path = "models/resnet18_CIFAR10_QAT/resnet18_int8_final.pth"

    model = build_int8_model(num_classes=10)

    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    model = torch.quantization.convert(model, inplace=False)

    state_dict = torch.load(int8_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print("INT8 model loaded.")

    fp32_model = convert_int8_to_fp32(model)

    torch.save(fp32_model.state_dict(), "resnet18_recovered_fp32.pth")
    print("FP32 (dequantized) model saved.")


if __name__ == "__main__":
    main()