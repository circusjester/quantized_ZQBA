python main_attack_int8.py ^
  --dataset Tiny ^
  --model-path models/resnet18_tiny_PTQ/resnet18_tiny_int8_ptq.pth ^
  --feature-maps-path feature_maps/resnet18_CIFAR10_dict.pickle ^
  --data-root-path ./data ^
  --impact-of-residual 0.1