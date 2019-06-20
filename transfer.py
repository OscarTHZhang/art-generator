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

def getFeatures(image, model, layers=None):
    """
    run a image forward and get features from the model
    """
    if layers is None:
        # match the layer number with the layer label as the paper states
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2', # content representation
            '28': 'conv5_1'
        }
    
    features = {}
    x = image
    for name, layer in model._modules.item():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x 
    
    return features 

def gramMatrix(tensor):
    """
    calculate the gram matrix of a given tensor
    """
    batchSize, depth, height, width = tensor.size()
    tensor = tensor.view(depth, height*width)
    res = torch.mm(tensor, tensor.t())
    return res 


