import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

def get_vgg19_model():
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return cnn, cnn_normalization_mean, cnn_normalization_std

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input
