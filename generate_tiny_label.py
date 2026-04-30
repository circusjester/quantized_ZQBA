import os
import csv
import torchvision
from pathlib import Path


def generate_labels_csv():
    # 数据集路径配置
    data_root = "./data/tiny-imagenet-200"
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    ann_file = os.path.join(val_dir, "val_annotations.txt")
    output_csv = os.path.join(val_dir, "labels_test.csv")

    print(f"正在读取训练集以获取类别映射 (class_to_idx)...")
    # 借用 ImageFolder 直接获取训练集中文件夹名称 (wnid) 到 数字标签 (0-199) 的映射
    trainset = torchvision.datasets.ImageFolder(root=train_dir)
    class_to_idx = trainset.class_to_idx

    print(f"共获取到 {len(class_to_idx)} 个类别的映射。")
    print(f"正在读取 {ann_file} 并生成 {output_csv}...")

    # 读取 val_annotations.txt 并写入 labels_test.csv
    count = 0
    with open(ann_file, "r") as f_in, open(output_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)

        # 写入表头 (如果你之前的 TinyImageNetDataset 需要表头，请保留这一行；如果不需要可以注释掉)
        # writer.writerow(["image_name", "label"])

        for line in f_in:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                img_name = parts[0]
                wnid = parts[1]

                # 将 wnid 转换为 0-199 的数字标签
                if wnid in class_to_idx:
                    label = class_to_idx[wnid]
                    writer.writerow([img_name, label])
                    count += 1
                else:
                    print(f"警告: 找不到 {wnid} 对应的标签！")

    print(f"生成完毕！共写入 {count} 条验证集数据到 {output_csv}")


if __name__ == "__main__":
    generate_labels_csv()