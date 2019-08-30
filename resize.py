from PIL import Image

from resizeimage import resizeimage

with open('style/raw/summer.jpg', 'rb') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [400, 300])
        cover.save('style/resize/summer_resize.jpg', image.format)

with open('content/raw/halifax.jpg', 'rb') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [400, 300])
        cover.save('content/resize/halifax_resize.jpg', image.format)
        
