import numpy as np
from pyexiv2 import Image
from matplotlib import pyplot as plt
from PIL import Image as im

imagepath = "...\\DJI_(1).JPG"
img = Image(imagepath)

List2D = np.array([[x, y] for y in range(100) for x in range(100)])
K = np.array([[1.95095996e+03, 0.00000000e+00, 8.04078979e+02],
              [0.00000000e+00, 1.94193005e+03, 6.19616028e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
R = np.array([[0.11094025, 0.99379022, -0.00855914],
              [0.99084571, -0.11126993, -0.07644462],
              [-0.07692229, 0., -0.99703709]])
t = np.array([[1.15294366e+07],
              [3.45719608e+06],
              [5.06170000e+02]])

def Get3Dfrom2D(List2D, K, R, t, d=1):
    # List2D : n x 2 array of pixel locations in an image
    # K : Intrinsic matrix for camera
    # R : Rotation matrix describing rotation of camera frame
    #     w.r.t world frame.
    # t : translation vector describing the translation of camera frame
    #     w.r.t world frame
    # [R t] combined is known as the Camera Pose.
    List3D = []
    List2D = np.array(List2D)
    # t.shape = (3,1)

    for p in List2D:
        # Homogeneous pixel coordinate
        p = np.array([p[0], p[1], 1]).T;
        p.shape = (3, 1)
        # print("pixel: \n", p)

        # Transform pixel in Camera coordinate frame
        pc = np.linalg.inv(K) @ p
        # print("pc : \n", pc, pc.shape)

        # Transform pixel in World coordinate frame
        pw = t + (R @ pc)
        # print("pw : \n", pw, t.shape, R.shape, pc.shape)

        # Transform camera origin in World coordinate frame
        cam = np.array([0, 0, 0]).T;
        cam.shape = (3, 1)
        cam_world = t + R @ cam
        # print("cam_world : \n", cam_world)

        # Find a ray from camera to 3d point
        vector = pw - cam_world
        unit_vector = vector / np.linalg.norm(vector)
        # print("unit_vector : \n", unit_vector)

        # Point scaled along this ray
        p3D = cam_world + d * unit_vector
        # print("p3D : \n", p3D)
        List3D.append(p3D)

    return List3D


def f(t):
    if type(t) == list or type(t) == tuple:
        return [f(i) for i in t]
    return t


List3D = Get3Dfrom2D(List2D, K, R, t, d=1)
x = []
y = []
z = []
for arr in List3D:
    x.append(arr[0])
    y.append(arr[1])
    z.append(arr[2])
img1 = im.open("F:\\img06\\JPG\\DJI_(1).JPG")
rgb_im = img1.convert('RGB')
value = rgb_im.getdata()
r = f(value)
r = np.array(r)
# r = []
# g = []
# b = []
# for y_pixel in range(100):
#     for x_pixel in range(100):
#         r_co, b_co, c_co = rgb_im.getpixel((y_pixel, x_pixel))
#         r.append(r_co/255)
#         g.append(b_co/255)
#         b.append(c_co/255)
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, alpha=1, color=r / 255, c='s')
ax.view_init(-120, 75)
plt.show()
