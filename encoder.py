import torch
import torch.nn as nn
from torchvision.models import resnet50

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Use pretrained weights
        resnet = resnet50(weights="IMAGENET1K_V1")  # Use pretrained weights
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
            features = features.squeeze(-1).squeeze(-1)
        features = self.linear(features)
        features = self.bn(features)
        return features
