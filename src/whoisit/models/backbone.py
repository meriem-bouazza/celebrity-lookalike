"""ResNet-18 from ImageNet, then Linear(512 -> 512) + BatchNorm1d as the embedding.

Cross-entropy and ArcFace both use this same embedding.
"""
import torch.nn as nn
from torchvision import models

EMBEDDING_SIZE = 512


class Backbone(nn.Module):
    def __init__(self, embedding_size=EMBEDDING_SIZE, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        # we do not want the 1000 ImageNet classes, so the last layer goes away
        features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.embedding = nn.Sequential(
            nn.Linear(features, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x):
        return self.embedding(self.resnet(x))
