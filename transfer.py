from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models

# load VGG model
vgg = models.vgg19(pretrained=True).features

# turn off the gradient computation
for param in vgg.parameters():
    param.requires_grad_(False)

# load the model to CPU / GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# print(vgg)

def loadImage(path, maxSize=400, shape=None):
    """
    Load in and transform an image, making sure the image is <= 400 px in 
    the x-y dimension
    """
    if "http" in path:
        # loading image from internet
        res = requests.get(path)
        image = Image.open(BytesIO(res.content)).convert("RGB")
    else:
        image = Image.open(path).convert("RGB")
    
    # sizing the image
    if max(image.size) > maxSize:
        size = maxSize
    else:
        size = max(image.size)
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image)[:3,:,:].unsqueeze(0)
    
    return image
