import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
import numpy as np
from scipy import interpolate, ndimage
from PIL import Image
import cv2
from pyexiv2 import Image as im2
import os
from pandas import *


path_of_the_directory = f"F:\\img16\\5\\New_TIF5"
ext = ('.jpg', '.JPG', '.TIF')
paths = []


for filename in os.scandir(path_of_the_directory):
    if filename.path.endswith(ext):
        f = os.path.join(path_of_the_directory, filename)
        if os.path.isfile(f):
            paths.append(f)

print(paths)
path_of_the_directory2 = f"F:\\img16\\5"
ext = ('.jpg', '.JPG', '.TIF')
paths2 = []

for filename in os.scandir(path_of_the_directory2):
    if filename.path.endswith(ext):
        f = os.path.join(path_of_the_directory, filename)
        if os.path.isfile(f):
            paths2.append(f)


data = read_csv("F:\\img05\\JPG\\output.csv")

x2 = data['GPSLongitude'].tolist()
y2 = data['GPSLatitude'].tolist()



def image_info(imagepath):

    img = im2(imagepath)
    exif = img.read_exif()
    # exifdata = piexif.load(imagepath)
    img.read_iptc()
    xmp = img.read_xmp()
    # b_name = xmp['Xmp.drone-dji.BandName']
    img.close()

    return xmp, exif

x = []
y = []

for images in paths2:
    img = im2(images)
    xmp = img.read_xmp()
    lat = float(xmp['Xmp.drone-dji.GpsLatitude'])
    lon = float(xmp['Xmp.drone-dji.GpsLongitude'])
    y.append(lat)
    x.append(lon)

print(x)
print(y)

# fig = plt.figure(figsize=(20, 14), dpi=72, facecolor='grey')
fig = plt.figure(figsize=(20, 14), dpi=100)
ax = fig.add_subplot(111)
ax.scatter(x, y)
ax.scatter(x2, y2, alpha=0)

for x0, y0, path, path2 in zip(x, y, paths, paths2):
    img2 = cv2.imread(path)
    PIL_image = Image.fromarray(img2).convert('RGBA')
    img = im2(path2)
    xmp = img.read_xmp()
    fyaw = float(xmp['Xmp.drone-dji.FlightYawDegree'])
    gyaw = float(xmp['Xmp.drone-dji.GimbalYawDegree'])
    yaw = (fyaw + gyaw) / 2
    image_arr = PIL_image.rotate(angle=-gyaw, expand=True, fillcolor=0)
    # 0.063 img06 0.046 img05 0.042 img02 0.031
    # adjust if image overlapping is wrong
    imagebox = OffsetImage(image_arr, zoom=0.0518)
    ab = AnnotationBbox(imagebox, (x0, y0), frameon=False)
    ax.add_artist(ab)

# for x, y, z in zip(x, y, paths):
#     ax.annotate('({})'.format(z), xy=(x, y), fontsize=15)
# for x, y in zip(x, y):
#     ax.annotate('({}, {})'.format(x, y), xy=(x, y), fontsize=22)
# label = "{:.5f}".format(x, y)
# ac = AnnotationBbox(label, xy)
# plt.annotate(label, (x,y), textcoord="offset points", xytext=(20,20), ha="center")
# plt.imshow(ax, origin='lower', extent=[-56.444444444, 56.444444444, -45.861111111, 45.861111111], aspect=1)
plt.axis('off')
plt.axis("equal")
fig.set_tight_layout(True)
plt.savefig(f'F:\\test\\9.7\\0907_TIF5(100).jpg', format='jpg')
plt.show()