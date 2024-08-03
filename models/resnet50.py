import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_resnet50_model():
    cnn = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    
    # Modify the model to fit style transfer requirements
    # Extract features from specific layers
    return cnn, cnn_normalization_mean, cnn_normalization_std

# Reuse ContentLoss and StyleLoss from vgg19.py
from models.vgg19 import ContentLoss, StyleLoss
