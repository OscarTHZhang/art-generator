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
print(device)
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
    
    # sizing the image if the image is too large to be processed
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

content = loadImage('content/resize/camila_resize.jpg').to(device)
style = loadImage('style/resize/mona_lisa_resize.jpg', shape=content.shape[-2:]).to(device)

def imgConvert(tensor):
    """
    convert tensor image to Numpy image for display
    """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    
    return image

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
ax1.imshow(imgConvert(content))
ax2.imshow(imgConvert(style))



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
    for name, layer in model._modules.items():
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

# get content and style features
content_features = getFeatures(content, vgg)
style_features = getFeatures(style, vgg)

# get the gram matrix for each layer
style_grams = {layer: gramMatrix(style_features[layer]) for layer in style_features}

target = content.clone().requires_grad_(True).to(device)

style_weights = {'conv1_1': 1.0,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

# alpha and beta
content_weight = 1 
style_weight = 1e6 

# updating target and calculating losses for each iteration

frequency = 400 # frequency of showing the transferring result

optimizer = optim.Adam([target], lr=0.003)
steps = 2000

for ii in range(1, steps + 1):
    
    target_features = getFeatures(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    
    style_loss = 0
    
    for layer in style_weights:
        
        target_feature = target_features[layer]
        target_gram = gramMatrix(target_feature)
        _, d, h, w = target_feature.shape
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_style_loss / (d * h * w)
        
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    # update target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if ii % frequency == 0:
        print('Total loss: ', total_loss.item())
        if ii == steps:
            ax3.imshow(imgConvert(target))

plt.figure()
plt.show()

