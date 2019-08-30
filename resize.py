from PIL import Image

from resizeimage import resizeimage


def resize_style(name, dimension=[400,300]):
    with open('style/raw/'+name, 'rb') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_cover(image, dimension)
            cover.save('style/resize/'+name.split('.')[0]+'_resize.'+name.split('.')[1], image.format)

def resize_content(name, dimension=[400,300]):
    with open('content/raw/'+name, 'rb') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_cover(image, dimension)
            cover.save('content/resize/'+name.split('.')[0]+'_resize.'+name.split('.')[1], image.format)
        

with open('style/raw/bright.jpg', 'rb') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [400, 300])
        cover.save('style/resize/bright_resize.jpg', image.format)

with open('content/raw/bridge.jpeg', 'rb') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [400, 300])
        cover.save('content/resize/bridge_resize.jpeg', image.format)
        
