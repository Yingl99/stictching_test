'''download earthpy_plot_revised put in the same folder'''
import rasterio
import earthpy_plot_revised as ep 
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os

# path_of_the_directory = f"F:\\test\\7.28\\0728_TIF3(1500).jpg"
# ext = ('.jpg', '.JPG', '.TIF')
# image_file1 = []
#
# for filename in os.scandir(path_of_the_directory):
#     if filename.path.endswith(ext):
#         f = os.path.join(path_of_the_directory, filename)
#         if os.path.isfile(f):
#             image_file1.append(f)
#
# path_of_the_directory = f"F:\\test\\7.28\\0728_TIF5(1500).jpg"
# ext = ('.jpg', '.JPG', '.TIF')
# image_file2 = []
#
# for filename in os.scandir(path_of_the_directory):
#     if filename.path.endswith(ext):
#         f = os.path.join(path_of_the_directory, filename)
#         if os.path.isfile(f):
#             image_file2.append(f)
image_file1 = ["F:\\test\\8.23\\0823_TIF3(1500).tif",]
image_file2 = ["F:\\test\\8.23\\0823_TIF5(1500)1.tif",]

for image1, image2 in zip(image_file1, image_file2):
    src1 = rasterio.open(image1)
    band_red = src1.read()
    # band_green = src1.read()

    src2 = rasterio.open(image2)
    band_nir = src2.read()


# Allow division by zero
    numpy.seterr(divide='ignore', invalid='ignore')

    band_nir = numpy.array(band_nir)
    # print(f"green:{band_green[0, 5151, 3790]},{band_green[1, 5151, 3790]},{band_green[2, 5151, 3790]}")
    band_red = numpy.array(band_red)
    # band_green = numpy.array(band_green)
    # print(f"nir:{band_nir[0, 5151, 3790]},{band_nir[1, 5151, 3790]},{band_nir[2, 5151, 3790]}")

    NDVI = (band_nir.astype(float) - band_red.astype(float)) / (band_nir + band_red)
    # green_index = (band_nir / band_green)*100 - 1
    # print(f"green_index:{green_index[0, 5151, 3790]},{green_index[1, 5151, 3790]},{green_index[2, 5151, 3790]}")

    fig1, ax = plt.subplots()
    # fig1.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig1 = plt.gcf()
    norm = plt.Normalize(1, 150)
    colorlist = ["whitesmoke", "lightgrey", "darkgrey", "whitesmoke", "white", "lightgrey", "darkgrey", "royalblue", "lime", "yellow", "orange", "red", "crimson"]
    # colorlist = ["whitesmoke", "whitesmoke", "whitesmoke", "whitesmoke", "whitesmoke", "yellowgreen", "yellowgreen", "yellow", "lime", "orange", "darkorange", "orangered", "red"]
    # colorlist = ["whitesmoke", "whitesmoke", "whitesmoke", "whitesmoke", "white", "whitesmoke", "whitesmoke", "royalblue", "lime", "yellow", "orange", "red", "crimson"]
    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist)
    ep.plot_bands(NDVI,
                  cmap=newcmp,
                  scale=False,
                  cbar=True,  # Change this to TRUE to display the color bar
                  ax=ax,
                  vmin=-1, vmax=1,
                  # cols=1,
                  save_or_not=True,
                  save_path="F:/test2/0823_NDVI3.jpg",
                  dpi_out=1500)
                  # bbox_inches_out="tight",
                  # pad_inches_out=0.1)
    # ep.plot_bands(green_index,
    #               cmap=newcmp,
    #               scale=False,
    #               cbar=True,  # Change this to TRUE to display the color bar
    #               ax=ax,
    #               vmin=-1, vmax=254,
    #               cols=1,
    #               save_or_not=True,
    #               save_path="F:/test2/0805_NDVI3.jpg",
    #               dpi_out=800,
    #               bbox_inches_out="tight",
    #               pad_inches_out=0.1)
    # plt.gca().set_axis_off()
    plt.show(block=False)
    plt.pause(1)
    plt.close("all")
    file_name = os.path.basename(image1)
    f_name, extension = os.path.splitext(file_name)
    # filename and where you save all the images
    # fig1.savefig('F:/test2/0823_NDVI{}.jpg'.format(f_name), dpi=2500, bbox_inches='tight', pad_inches=0.0)
