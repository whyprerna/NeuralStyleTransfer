import torch
import torchvision.transforms as transforms
from PIL import Image

def load_image(image_file):
    img = Image.open(image_file)
    return img

def process_image(img, imsize):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()])
    image = loader(img).unsqueeze(0)
    return image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.float)

def imshow(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image