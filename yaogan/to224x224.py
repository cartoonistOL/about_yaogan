import os
import shutil

from osgeo import gdal


root = r'F:\文件\combine_siji_vienna_ex'

print(os.listdir(root))
for date in os.listdir(root):
    datepath = root + "\\" + date
    print(date)
    for file in os.listdir(datepath):
        path = os.path.join(datepath, file)
        dataset = gdal.Open(path)
        hang_1, lie_1 = [0, 0]
        hang_2, lie_2 = [224, 224]
        target =r'F:\文件\pretrain' + '\\' + date + '\\' + file
        src: gdal.Dataset = dataset
        gdal.Translate(target, src, srcWin=[lie_1, hang_1, lie_2 - lie_1, hang_2 - hang_1],
                       options=['-a_scale', '1'])
        print("{}".format(target))
        #shutil.move(target, r'F:\文件\pretrain'+'\\' + date + '\\' + file)
