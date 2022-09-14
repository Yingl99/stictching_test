from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = 1000000000

'''image from test image to csv with pixel colour value'''
img = Image.open('F:\\test\\test_RGB3.jpg')
print('Inspect a few pixels in the original image:')
for y in np.arange(3):
    for x in np.arange(3):
        print(x, y, img.getpixel((x, y)))

# Modified for RGB images from: https://stackoverflow.com/a/60783743/11089932
img_np = np.array(img)
print(img_np.shape[:2])
xy_coords = np.flip(np.column_stack(np.where(np.all(img_np >= 0, axis=2))), axis=1)
rgb = np.reshape(img_np, (np.prod(img_np.shape[:2]), 3))

# Add pixel numbers in front
pixel_numbers = np.expand_dims(np.arange(1, xy_coords.shape[0] + 1), axis=1)
print(pixel_numbers.size, xy_coords.size, rgb.size)
value = np.hstack([pixel_numbers, xy_coords, rgb])
print('\nCompare pixels in result:')
for y in np.arange(3):
    for x in np.arange(3):
        print(value[(value[:, 1] == x) & (value[:, 2] == y)])

# Properly save as CSV
np.savetxt("F:\\test\\outputdata2.csv", value, delimiter=',', fmt='%4d')