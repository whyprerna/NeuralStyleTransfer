import torch
import torch.nn as nn
import torch.optim as optim
from models.vgg19 import get_vgg19_model, ContentLoss, StyleLoss
from models.alexnet import get_alexnet_model
from models.resnet50 import get_resnet50_model

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.std = torch.tensor(std).view(-1, 1, 1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, img):
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, content_layers, style_layers):
    normalization = Normalization(normalization_mean, normalization_std).to(style_img.device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # Increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def run_style_transfer(model_name, content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    if model_name == 'vgg19':
        cnn, cnn_normalization_mean, cnn_normalization_std = get_vgg19_model()
        content_layers_default = ['conv_4']
        style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    elif model_name == 'alexnet':
        cnn, cnn_normalization_mean, cnn_normalization_std = get_alexnet_model()
        content_layers_default = ['conv_2']
        style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    elif model_name == 'resnet50':
        cnn, cnn_normalization_mean, cnn_normalization_std = get_resnet50_model()
        content_layers_default = ['layer4']
        style_layers_default = ['layer1', 'layer2', 'layer3', 'layer4']
    else:
        raise ValueError("Invalid model name")

    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     cnn_normalization_mean, cnn_normalization_std,
                                                                     style_img, content_img,
                                                                     content_layers_default, style_layers_default)

    optimizer = optim.LBFGS([input_img.requires_grad_()])

    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            loss = style_score * style_weight + content_score * content_weight
            loss.backward()

            run[0] += 1
            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img
