set FEATURE_PATH=feature_maps\resnet18_CIFAR10_dict.pickle
set MODEL_TYPE=ResNet18
set MODEL_PATH=models\resnet18_tiny\resnet18_tiny_best.pth
set DATA_PATH=data
set DATASET=Tiny

python main_attack.py ^
  --feature-maps-path %FEATURE_PATH% ^
  --model-type %MODEL_TYPE% ^
  --model-path %MODEL_PATH% ^
  --dataset %DATASET% ^
  --data-root-path %DATA_PATH% ^
  --impact-of-residual 0.1