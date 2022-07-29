import cv2
import numpy as np
from pyexiv2 import Image
import os


def image_info(imagepath):
    """obtain xmp、exif data"""

    img = Image(imagepath)
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


def raw2ref(xmp, img):
    """
    DN value for reflectance calculation
    """

    blacklevel = exif['Exif.Image.BlackLevel']
    blacklevel = int(blacklevel)

    sensorgain = float(xmp['Xmp.drone-dji.SensorGain'])

    ExposureTime = float(xmp['Xmp.drone-dji.ExposureTime'])

    CalibratedOpticalCenterX = float(xmp['Xmp.drone-dji.CalibratedOpticalCenterX'])
    CalibratedOpticalCenterY = float(xmp['Xmp.drone-dji.CalibratedOpticalCenterY'])

    VignettingData = xmp['Xmp.drone-dji.VignettingData']
    # '0.000218235, 1.20722e-6, -2.8676e-9, 5.1742e-12, -4.16853e-15, 1.36962e-18'
    VignettingData = VignettingData.split(",")
    VignettingData = [float(i) for i in VignettingData]

    pcamera_band = float(xmp['Xmp.drone-dji.SensorGainAdjustment'])

    Camera_Irradiance = float(xmp['Xmp.Camera.Irradiance'])

    Band_canmera = np.zeros((h, w))

    img = (img - blacklevel) / 65535

    for i in range(h):
        for j in range(w):

            r = ((j - CalibratedOpticalCenterX) ** 2 + (i - CalibratedOpticalCenterY) ** 2) ** 0.5

            correction = ((VignettingData[5]) * (r ** 6) + (VignettingData[4]) * (r ** 5) + (VignettingData[3]) * (
                    r ** 4) + (VignettingData[2]) * (r ** 3) + (VignettingData[1]) * (r ** 2)
                          + (VignettingData[0]) * r) + 1.0

            Band_canmera[i, j] = (img[i, j] * correction) / (sensorgain * ExposureTime / 1000000.0)

    Band_ref = (Band_canmera * pcamera_band) / Camera_Irradiance

    return Band_ref


def dis_correction(cam_mat, dist_coeffs, Band_ref):
    ''' camera distortion correction '''

    (h, w) = img.shape[:2]

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_mat, dist_coeffs, (w, h), 0, (w, h))

    # dst = cv2.undistort(Band_ref, cam_mat, dist_coeffs, None, newcameramtx)
    mapx, mapy = cv2.initUndistortRectifyMap(cam_mat, dist_coeffs, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(Band_ref, mapx, mapy, cv2.INTER_LINEAR)

    print("newcameramtx:", newcameramtx)

    x, y, w, h = roi
    print(x, y, w, h)
    dst = dst[y:y + h, x:x + w]

    return dst


if __name__ == '__main__':
    # change the path to where the TIF image is located at (applicable to TIF1-5)
    path_of_the_directory = "F:\\6.9\\1"
    ext = ('.jpg', '.JPG', '.TIF')
    imagepath = []

    for filename in os.scandir(path_of_the_directory):
        if filename.path.endswith(ext):
            f = os.path.join(path_of_the_directory, filename)
            if os.path.isfile(f):
                imagepath.append(f)

    for image1 in imagepath:
        file_name = os.path.basename(image1)
        f_name, extension = os.path.splitext(file_name)
        # change the path of the folder to where you want to save
        outimage = "F:/6.9/1/New_TIF1/" + f_name + ".jpg"

        xmp, exif = image_info(image1)

        img = cv2.imread(image1, 2)

        h, w = img.shape

        cam_mat, dist_coeffs = image_mat(xmp)

        Band_ref = raw2ref(xmp, img)

        print(Band_ref[0, 1])
        print(h, w)

        new_image = dis_correction(cam_mat, dist_coeffs, Band_ref)

        cv2.imwrite(outimage, new_image * 100000 * 0.00314949)

# 计算误差
# tot_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#     tot_error += error
# print ("total error: ", tot_error/len(objpoints))
