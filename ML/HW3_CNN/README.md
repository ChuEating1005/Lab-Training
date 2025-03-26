# ML HW3 - CNN
> [!Note]
> Task: Image Classification

## Dataset
The images are collected from the food-11 dataset splitted into 11 classes.
- Training set: 10000 labeled images
- Validation set: 3643 labeled images
- Testing set: 3000 images without labeled

### Download Link
https://www.kaggle.com/competitions/ml2023spring-hw3/data
```bash
unzip food11.zip
```

## Environment
- Python 3.10.6
- OS: Ubuntu 24.04.2 LTS
- GPU: NVIDIA GeForce RTX 4090
### Packages
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia scikit-learn
```
### Open Tensorboard
```bash
tensorboard --logdir=runs/
```

## Result
| Model | Private Score | Public Score | Parameters |
|-------|---------------|--------------|------------|
| Cross-validation (Refined-CNN, Hard voting) | 0.8513 | 0.84733 | - |
| Cross-validation (Refined-CNN, Soft voting) | 0.8600 | 0.84666 | - |
| Default (Refined-CNN with dropout & batch-norm) | 0.8220 | 0.81533 | - |
| Default (Original-CNN) | 0.72266 | 0.72600 | - |
| EfficientNet-b0 | 0.7561 | - | 5.3M |
| ResNet18 | 0.7371 | - | 11.7M |
| ResNet50 | 0.7065 | - | 25.6M |
