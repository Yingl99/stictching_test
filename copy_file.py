"""   copy xxx0.JPG, xxx1.TIF, xxx2.TIF, xxx3.TIF, xxx4.TIF, xxx5.TIF each type of photo to another folder
      * must exclude reflectance panel photo (标定板）                     """

import os
import shutil
from natsort import natsorted

# change to the folder that includes all flight photo (eg. 140FPLAN, 141FPLAN, 142FPLAN)
src = r"F:/6.9/6.9"
num = range(0, 6)
count = 1

for root, subdirectories, files in os.walk(src):
    for subdirectory in subdirectories:
        path = os.path.join(root, subdirectory)
        for f in os.listdir(path):
            f2, ext = os.path.splitext(f)
            for n in num:
                if f2.endswith(str(n)):
                    outpath = f"F:\\6.9\\{n}"
                    shutil.copyfile(os.path.join(path, f), os.path.join(outpath, f))
                    new_file_name = os.path.join(outpath, "DJI_(" + str(count) + ")" + ext)
                    os.rename(os.path.join(outpath, f), new_file_name)
                    count += 1

for n in num:
    i = 1
    # change to the filename that you've created
    outpath = f"F:\\6.9\\{n}"
    for f1 in natsorted(os.listdir(outpath)):
        f22, ext2 = os.path.splitext(f1)
        new_file_name = os.path.join(outpath, "DJI_(" + str(i) + ")" + ext2)
        os.rename(os.path.join(outpath, f1), new_file_name)
        i += 1
