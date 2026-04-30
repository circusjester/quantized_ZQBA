set FEATURE_PATH=resnet18_CIFAR10_dict.pickle
set MODEL_TYPE=ResNet18
set MODEL_PATH=models\resnet18_CIFAR10\resnet18_cifar10_best.pth
set DATA_PATH=data
set DATASET=CIFAR10

python get_feature_maps.py ^
  --feature-maps-path %FEATURE_PATH% ^
  --model-type %MODEL_TYPE% ^
  --model-path %MODEL_PATH% ^
  --dataset %DATASET% ^
  --data-root-path %DATA_PATH%