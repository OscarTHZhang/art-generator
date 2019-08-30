from PIL import Image

from resizeimage import resizeimage

# with open('style/raw/starry_night.jpg', 'rb') as f:
#     with Image.open(f) as image:
#         cover = resizeimage.resize_cover(image, [400, 200])
#         cover.save('style/resize/starry_night_resize.jpg', image.format)

with open('content/raw/mendota_deck.jpg', 'rb') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [400, 200])
        cover.save('content/resize/mendota_deck_resize.jpg', image.format)
        
