import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
import numpy as np
from PIL import Image
import cv2
from pyexiv2 import Image as im2
import os
from pandas import *

# path of the images for image stitching
path_of_the_directory = "F:\\img11\\0"
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


fig = plt.figure(figsize=(20, 14), dpi=72)
ax = fig.add_subplot(111)
ax.scatter(x, y)
ax.scatter(x2, y2, alpha=0)


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
    imagebox = OffsetImage(image_arr, zoom=0.043)
    ab = AnnotationBbox(imagebox, (x0, y0), frameon=False)
    ax.add_artist(ab)

''' lon4 = longitude in decimal (converted from degree) from excel file, lat4 = latitude, Chl4 = Chlorophyll content '''
df = pd.read_csv('C:/LocationShip.csv')
im = ax.scatter(df.Lon4, df.Lat4, c=df.Chl4, cmap='plasma', s=10, zorder=3)
fig.colorbar(im, ax=ax)
# for x, y, z in zip(x, y, paths):
#     ax.annotate('({})'.format(z), xy=(x, y), fontsize=15)
# for x, y in zip(x, y):
#     ax.annotate('({}, {})'.format(x, y), xy=(x, y), fontsize=22)
# label = "{:.5f}".format(x, y)
# ac = AnnotationBbox(label, xy)
# plt.annotate(label, (x,y), textcoord="offset points", xytext=(20,20), ha="center")
# plt.imshow(ax, origin='lower', extent=[-56.444444444, 56.444444444, -45.861111111, 45.861111111], aspect=1)
# plt.axis('off')
plt.axis("equal")
# the filename and where you save the file
# the dpi of the image can be changed as well according the the requirement
plt.savefig('F:\\test\\0805_RGB2.jpg', dpi=300, format='jpg', transparent=True, bbox_inches='tight')
plt.show()

