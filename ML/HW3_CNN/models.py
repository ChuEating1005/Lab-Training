import torch.nn as nn
from torchvision import models

def get_model(model_name):
    if model_name == 'default':
        model = DefaultClassifier()
        _exp_name = "default"
    elif model_name == 'resnet':
        model = ResNet18Classifier()
        _exp_name = "resnet18"
    elif model_name == 'mobilenet':
        model = MobileNetV2Classifier()
        _exp_name = "mobilenet_v2"
    elif model_name == 'efficientnet':
        model = EfficientNetB0Classifier()
        _exp_name = "efficientnet_b0"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, _exp_name

class DefaultClassifier(nn.Module):
    def __init__(self):
        super(DefaultClassifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input dimension [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5), 
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5), 

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), 

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class MobileNetV2Classifier(nn.Module):
    def __init__(self):
        super(MobileNetV2Classifier, self).__init__()
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 11)

    def forward(self, x):
        return self.model(x)

class ResNet18Classifier(nn.Module):
    def __init__(self):
        super(ResNet18Classifier, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 11)

    def forward(self, x):
        return self.model(x)

class EfficientNetB0Classifier(nn.Module):
    def __init__(self):
        super(EfficientNetB0Classifier, self).__init__()
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 11)

    def forward(self, x):
        return self.model(x)