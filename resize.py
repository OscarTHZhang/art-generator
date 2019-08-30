from PIL import Image

from resizeimage import resizeimage

with open('style/raw/bright.jpg', 'rb') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [400, 300])
        cover.save('style/resize/bright_resize.jpg', image.format)

with open('content/raw/bridge.jpeg', 'rb') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [400, 300])
        cover.save('content/resize/bridge_resize.jpeg', image.format)
        
