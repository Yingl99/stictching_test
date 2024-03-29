import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
import numpy as np
from PIL import Image
import cv2
from pyexiv2 import Image as im2
import os
from pandas import *
import csv
from itertools import zip_longest

# path of the images for image stitching
path_of_the_directory = "F:\\img16\\0"
ext = ('.jpg', '.JPG', '.TIF')
paths = []

for filename in os.scandir(path_of_the_directory):
    if filename.path.endswith(ext):
        f = os.path.join(path_of_the_directory, filename)
        if os.path.isfile(f):
            paths.append(f)

# a .csv file contain of 6.21 gps latitude and longitude
# (downloadable at https://github.com/Yingl99/stictching_test/blob/main/output.csv)
data = read_csv("F:\\img05\\JPG\\output.csv")

x2 = data['GPSLongitude'].tolist()
y2 = data['GPSLatitude'].tolist()


def image_info(imagepath):
    """obtain xmp、exif data"""

    img = im2(imagepath)
    exif = img.read_exif()
    # exifdata = piexif.load(imagepath)
    img.read_iptc()
    xmp = img.read_xmp()
    # b_name = xmp['Xmp.drone-dji.BandName']
    img.close()

    return xmp, exif


def image_mat(xmp):
    """read metadata from xmp, exif data"""

    clibrate_data = xmp['Xmp.drone-dji.DewarpData']
    clibrate_data = clibrate_data.split(";")[1].split(",")

    fx = float(clibrate_data[0])
    fy = float(clibrate_data[1])
    cX = float(clibrate_data[2]) + float(xmp['Xmp.drone-dji.CalibratedOpticalCenterX'])
    cY = float(clibrate_data[3]) + float(xmp['Xmp.drone-dji.CalibratedOpticalCenterY'])

    cam_mat = np.zeros((3, 3))
    cam_mat[0, 0] = fx
    cam_mat[1, 1] = fy
    cam_mat[2, 2] = 1.0
    cam_mat[0, 2] = cX
    cam_mat[1, 2] = cY
    print(cam_mat)

    k1 = float(clibrate_data[4])
    k2 = float(clibrate_data[5])
    p1 = float(clibrate_data[6])
    p2 = float(clibrate_data[7])
    k3 = float(clibrate_data[8])

    dist_coeffs = np.array([k1, k2, p1, p2, k3]).reshape((1, 5))
    # dist_coeffs = (k1, k2, p1, p2)
    # dist_coeffs = np.array([k1, k2, p1, p2, k3])
    print(dist_coeffs)

    return cam_mat, dist_coeffs

x = []
y = []

for images in paths:
    img = im2(images)
    xmp = img.read_xmp()
    lat = float(xmp['Xmp.drone-dji.GpsLatitude'])
    lon = float(xmp['Xmp.drone-dji.GpsLongitude'])
    y.append(lat)
    x.append(lon)

print(x)
print(y)


fig = plt.figure(figsize=(20, 14), dpi=100)
ax = fig.add_subplot(111)
ax.scatter(x, y)
ax.scatter(x2, y2, alpha=0)
ax.scatter(120.85573312, 31.06219546, zorder=4)


xmp, exif = image_info(paths[10])

cam_mat, dist_coeffs = image_mat(xmp)

for x0, y0, path in zip(x, y, paths):
    img2 = cv2.imread(path)
    h, w = img2.shape[0], img2.shape[1]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_mat, dist_coeffs, (w, h), 0, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(cam_mat, dist_coeffs, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img2, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(dst).convert('RGBA')
    img = im2(path)
    xmp = img.read_xmp()
    fyaw = float(xmp['Xmp.drone-dji.FlightYawDegree'])
    gyaw = float(xmp['Xmp.drone-dji.GimbalYawDegree'])
    yaw = (fyaw + gyaw) / 2
    # gyaw can be negative/positive based on the result (if noticed the image is not rotated correctly)
    image_arr = PIL_image.rotate(angle=-gyaw, expand=True, fillcolor=0)
    #  the value of zoom can be change according to the overlapping ratios
    #  for e.g. 0.043 for 6.21 images, 0.031 for 5.27 images
    #  adjust based on the result
    imagebox = OffsetImage(image_arr, zoom=0.0518)
    ab = AnnotationBbox(imagebox, (x0, y0), frameon=False)
    ax.add_artist(ab)

# df = pd.read_csv('C:/LocationShip.csv')
# im = ax.scatter(df.Lon4, df.Lat4, c=df.Chl4, cmap='plasma', s=10, zorder=3)
plt.axis('off')
plt.axis("equal")
fig.set_tight_layout(True)
plt.gca().invert_yaxis()
''' the filename and where you save the file
    the dpi of the image can be changed as well according to the requirement'''
# plt.savefig('F:\\test\\test_RGB2.jpg', format='jpg')
fig.canvas.draw()
# plt.show()
# 309, 831 ; 348, 902
'''input the file path of the image from testimage'''
img = Image.open('F:\\test\\test_RGB3.jpg')
pixel_coordinate_x = []
pixel_coordinate_y = []
gps_coordinate_x = []
gps_coordinate_y = []
w, h = img.size
print(w, h)
for y in range(h):
    for x in range(w):
        print(x, y)
        pixel_coordinate_x.append(x)
        pixel_coordinate_y.append(y)
        coordinates = x, y
        point = ax.transData.inverted().transform(coordinates)
        gps_coordinate_x.append(point[0])
        gps_coordinate_y.append(point[1])
        print(point)

print(gps_coordinate_x)
print(gps_coordinate_y)

gps = zip(pixel_coordinate_x,pixel_coordinate_y, gps_coordinate_x, gps_coordinate_y)
with open('F:\\test\\Book2.csv', 'w', encoding="ISO-8859-1", newline='') as s:
    w = csv.writer(s)
    for row in gps:
        w.writerow(row)
''' check point from image 
    select point from image and insert pixel coordinate to x
    ax.transData.inverted().transform(x) is for pixel to GPS coordinate
    ax.transData.transform(x) is for gps to pixel coordinate'''
# x = 309, 831
# point = ax.transData.inverted().transform(x)
# print(point)
# x = 348, 902
# point = ax.transData.inverted().transform(x)
# print(point)
# x = 120.8557222, 31.06280556
# point = ax.transData.transform(x)
# print(point)
# x = 120.8570833, 31.06033333
# point = ax.transData.transform(x)
# print(point)