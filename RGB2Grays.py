from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

im = 'F:\\test\\6.21\\0621_TIF5.jpg'
img = Image.open(im).convert('L')
img.save('F:\\test\\6.21\\0621_TIF5.tif')