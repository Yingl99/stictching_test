'''download gdal_merge.py and put it in the same folder'''
import subprocess
merge_command = ["python", "gdal_merge.py", "-o", "F:\\test\\6.21\\1.tif", "F:\\test\\6.21\\0621_TIF1.tif", "F:\\test\\6.21\\0621_TIF2.tif", "F:\\test\\6.21\\0621_TIF3.tif", "F:\\test\\6.21\\0621_TIF4.tif", "F:\\test\\6.21\\0621_TIF5.tif"]
subprocess.call(merge_command)