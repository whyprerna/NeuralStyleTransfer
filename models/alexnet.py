import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights

def get_alexnet_model():
    cnn = alexnet(weights=AlexNet_Weights.DEFAULT).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return cnn, cnn_normalization_mean, cnn_normalization_std

# Reuse ContentLoss and StyleLoss from vgg19.py
from models.vgg19 import ContentLoss, StyleLoss
