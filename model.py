import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

# 定义模型
def get_model(num_classes=3, pretrained=True):
    model = resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model