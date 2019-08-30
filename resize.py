from PIL import Image

from resizeimage import resizeimage

with open('style/raw/mona_lisa.jpg', 'rb') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [200, 400])
        cover.save('style/resize/mona_lisa_resize.jpg', image.format)

with open('content/raw/camila.jpg', 'rb') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [200, 400])
        cover.save('content/resize/camila_resize.jpg', image.format)
        
