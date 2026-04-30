
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn

import torch.nn.functional as F

import pickle
import random
import argparse
import os

from tqdm import tqdm
from Tiny_dataset_loader import TinyImageNetDataset
from wideresnet import WideResNet

def load_pickle(orig_file):
    with open(orig_file, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_


def get_residual_image(feature_maps_dict):
    feature_map_name = random.choice(list(feature_maps_dict.keys()))
    feature_map = feature_maps_dict[feature_map_name]

    return feature_map


def single_run_attack(inputs, img_count, dataset, feature_maps_dict):
    c, h, w = inputs[0].shape
    adv_images = torch.empty((0, c, h, w)).to('cuda')

    for i in range(len(inputs)):
        image = inputs[i]
        res_img = get_residual_image(feature_maps_dict)

        gb_tensor = transforms.ToTensor()(res_img).to('cuda')
        # possible interpolation
        gb_h, gb_w = gb_tensor.shape[1], gb_tensor.shape[2]
        if gb_h != h or gb_w != w:
            # the expected input of F.interpolate is a 4D tensor (Batch, Channel, Height, Width)
            # unsqueeze(0) to expand the dimension of Batch
            gb_tensor = gb_tensor.unsqueeze(0)
            # bilinear interpolation
            gb_tensor = F.interpolate(gb_tensor, size=(h, w), mode='bilinear', align_corners=False)
            # squeeze(0) to remove Batch
            gb_tensor = gb_tensor.squeeze(0)

        adv_image = torch.add(image, gb_tensor, alpha=impact_of_residual)
        adv_image = adv_image / torch.max(adv_image)
        adv_image = torch.unsqueeze(adv_image, 0)

        adv_images = torch.cat((adv_images, adv_image), 0)
        img_count += 1

    return img_count, adv_images


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-maps-path', type=str, default='feature_maps_CIFAR10/feature_maps_resnet18_dict.pickle', help='Path to the pickle containing the feature maps')
    parser.add_argument('--model-type', type=str, default='ResNet18', help='Model architecture to evaluate')
    parser.add_argument('--model-path', type=str, default='../Adversarial_Attack/models/resnet18_CIFAR10_not_attacked/checkpoint_best.pth', help='Path to the previously trained model to evaluate')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to evaluate')
    parser.add_argument('--data-root-path', type=str, default='/home/joana/Adversarial_Attack/datasets', help='Path to the data directory')
    parser.add_argument("--impact-of-residual", type=float, default=0.4, help="Alpha used in clean + alpha * residual")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    global impact_of_residual
    args = get_args()

    batch_size = 1
    impact_of_residual = args.impact_of_residual

    dataset = args.dataset
    data_root_path = args.data_root_path
    
    model_type = args.model_type
    model_path = args.model_path

    feature_maps_path = args.feature_maps_path
    feature_maps_dict = load_pickle(feature_maps_path)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])

    # adjustment for Tiny
    if dataset == 'Tiny':
        # 224x224 for Tiny ImageNet and normalization
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        # for other datasets
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    if dataset == 'CIFAR10':
        num_labels = 10
        testset = torchvision.datasets.CIFAR10(root=data_root_path, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    elif dataset == 'CIFAR100':
        num_labels = 100
        testset = torchvision.datasets.CIFAR100(root=data_root_path, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    elif dataset == 'Tiny':
        num_labels = 200
        data_path = os.path.join(data_root_path, 'tiny-imagenet-200')
        # testset_path = os.path.join(data_path, 'val')
        temp_path = os.path.join(data_path, 'val')
        testset_path = os.path.join(temp_path, 'images')
        # test_csv_path = os.path.join(testset_path, 'labels_test.csv')
        test_csv_path = os.path.join(temp_path, 'labels_test.csv')

        testset = TinyImageNetDataset(csv_file=test_csv_path, root_dir=testset_path, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    else:
        assert False

    if model_type == 'ResNet18':
        model = torchvision.models.resnet18()
        if dataset in ['CIFAR10', 'CIFAR100']:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_labels)

    elif model_type == 'ResNet50':
        model = torchvision.models.resnet50()
        if dataset in ['CIFAR10', 'CIFAR100']:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_labels)

    elif model_type == 'ResNet101':
        model = torchvision.models.resnet101()
        if dataset in ['CIFAR10', 'CIFAR100']:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_labels)
    
    elif model_type == 'MobileNetv2':
        model = torchvision.models.mobilenet_v2(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_labels)

    elif model_type == 'VGG19':
        model = torchvision.models.vgg19(pretrained=False)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_labels)

    elif model_type == 'EfficientNetB2':
        model = torchvision.models.efficientnet_b2(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_labels)

    elif model_type == 'WideResNet28_10':
        model = WideResNet(depth=28, num_classes=num_labels, widen_factor=10)

    else:
        assert False

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to('cuda')
    model.eval()

    img_count = 0
    total_lbls = 0
    clean_correct = 0
    attack_correct = 0
    counter = 0
    for i, data in enumerate(tqdm(test_loader), 0):
        inputs, labels = data
        
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')

        clean_outputs = model(inputs)
        _, predicted = torch.max(clean_outputs, 1)
        ind_to_fool = (predicted == labels).nonzero().squeeze().to('cuda')
        # clean_correct += (clean_outputs == labels).sum().item()
        clean_correct += (predicted == labels).sum().item()

        if len(ind_to_fool.shape) == 0:
            ind_to_fool = ind_to_fool.unsqueeze(0)

        if ind_to_fool.numel() != 0:
            x_to_fool = inputs[ind_to_fool].to('cuda')
            y_to_fool = predicted[ind_to_fool]
            
            img_count, adv_imgs = single_run_attack(x_to_fool, img_count, dataset, feature_maps_dict)
            adv_imgs = adv_imgs.float().to('cuda')
            lbls = y_to_fool.to('cuda')

            inputs_metrics = inputs[0].cpu().detach().numpy()
            adv_metrics = adv_imgs[0].cpu().detach().numpy()

            outputs = model(adv_imgs)
            _, adv_predicted = torch.max(outputs, 1)
            attack_correct += (adv_predicted == lbls).sum().item()

            # --- 新增：仅抓取第一张图片进行可视化和 L-inf 评估 ---
            if counter == 0:
                import torchvision.utils as vutils

                # 定义反归一化 (Denormalization)
                # 如果是 Tiny 数据集，需要根据之前 transform 的均值和方差还原到 0-1
                if dataset == 'Tiny':
                    inv_normalize = transforms.Normalize(
                        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                    )
                    orig_vis = inv_normalize(x_to_fool[0].cpu())
                    adv_vis = inv_normalize(adv_imgs[0].cpu())
                else:
                    # 如果是 CIFAR10 且没做 Normalize，直接取值即可
                    orig_vis = x_to_fool[0].cpu()
                    adv_vis = adv_imgs[0].cpu()

                # 强制截断到 0-1 范围
                orig_vis = torch.clamp(orig_vis, 0, 1)
                adv_vis = torch.clamp(adv_vis, 0, 1)

                # 计算 L-infinity Norm (最大像素偏差)
                l_inf = torch.max(torch.abs(orig_vis - adv_vis)).item()
                print(f"\n\n[监控] FP32 抓取成功！L-infinity 最大扰动: {l_inf:.4f}")

                # 拼接并保存图像 (左：原图，右：对抗样本)
                comparison = torch.cat([orig_vis, adv_vis], dim=2)
                vutils.save_image(comparison, "fp32_adversarial_eyeball_test.png")
                print("[监控] 对比图已保存至 fp32_adversarial_eyeball_test.png，请查看！\n")

                counter += 1  # 确保只保存一次
            # ----------------------------------------------------

        else:
            inputs_metrics = inputs[0].cpu().detach().numpy()
            
        total_lbls += labels.size(0)

    accuracy = 100.0 * clean_correct / total_lbls
    print('Clean Accuracy {:.2f}'.format(accuracy))
    
    attack_accuracy = 100.0 * attack_correct / total_lbls
    print('Attack Accuracy {:.2f}'.format(attack_accuracy))