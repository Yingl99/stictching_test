import rasterio
import earthpy.plot as ep
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os

path_of_the_directory = f"F:\\img02\\img02\\TIF3\\New_TIF3"
ext = ('.jpg', '.JPG', '.TIF')
image_file1 = []

for filename in os.scandir(path_of_the_directory):
    if filename.path.endswith(ext):
        f = os.path.join(path_of_the_directory, filename)
        if os.path.isfile(f):
            image_file1.append(f)

path_of_the_directory = f"F:\\img02\\img02\\TIF5\\New_TIF5"
ext = ('.jpg', '.JPG', '.TIF')
image_file2 = []

for filename in os.scandir(path_of_the_directory):
    if filename.path.endswith(ext):
        f = os.path.join(path_of_the_directory, filename)
        if os.path.isfile(f):
            image_file2.append(f)

for image1, image2 in zip(image_file1, image_file2):
    src1 = rasterio.open(image1)
    band_red = src1.read()
    # print(band_red)

    src2 = rasterio.open(image2)
    band_nir = src2.read()
    # print(band_nir)

# Allow division by zero
    numpy.seterr(divide='ignore', invalid='ignore')

    band_nir = numpy.array(band_nir)
    band_red = numpy.array(band_red)

    NDVI = (band_nir.astype(float) - band_red.astype(float)) / (band_nir + band_red)

    fig1, ax = plt.subplots(1)
    fig1.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig1 = plt.gcf()
    norm = plt.Normalize(1, 150)
    colorlist = ["whitesmoke", "lightgrey", "darkgrey", "whitesmoke", "white", "lightgrey", "darkgrey", "royalblue", "lime", "yellow", "orange", "red", "crimson"]
    # colorlist = ["whitesmoke", "whitesmoke", "whitesmoke", "whitesmoke", "white", "whitesmoke", "whitesmoke", "royalblue", "lime", "yellow", "orange", "red", "crimson"]
    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)
    ep.plot_bands(NDVI,
                  cmap=newcmp,
                  scale=False,
                  cbar=False,  # Change this to TRUE to display the color bar
                  ax=ax,
                  vmin=-1, vmax=1)
    plt.gca().set_axis_off()
    plt.show(block=False)
    plt.pause(1)
    plt.close("all")
    file_name = os.path.basename(image1)
    f_name, extension = os.path.splitext(file_name)
    # filename and where you save all the images
    fig1.savefig('F:/test2/img02_NDVI/{}.jpg'.format(f_name), dpi=200, bbox_inches='tight', pad_inches=0.0)
