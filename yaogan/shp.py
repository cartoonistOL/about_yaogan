import numpy as np
# gdal用来处理栅格数据
from osgeo import gdal
# ogr用来处理矢量数据
from osgeo import ogr
import os
from os.path import join
import os
os.environ['PROJ_LIB'] = r'C:\Users\owl\anaconda3\envs\pythonProject\Library\share\proj'


# 矢量以及栅格保存路径
shp_root = r'C:\Users\owl\Desktop\planet_116.19_39.753_66eefaf9-shp (1)\shape\landuse.shp'
path_root = r'C:\Users\owl\Desktop\20201209\LC08_L1TP_123032_20201209_20210313_02_T1_B3.TIF'
save_root = r'C:\Users\owl\Desktop\123'

if not os.path.exists(save_root):
    os.makedirs(save_root)

# 定义栅格的读取
gdal.AllRegister()
def ImgOpen(dir):
    data = gdal.Open(dir, gdal.GA_ReadOnly)
    if data == 'None':
        print('图像无法加载')
    return data
# 根据路径判断尾缀
path_names = sorted(os.listdir(path_root))
# 设置矢量及栅格数据的保存位置
img_path = []
# 判断文件尾缀，保存到对应的矢量或栅格列表中
for path_name in path_names:
    path_namefor = os.path.splitext(path_name)[0]
    path_nameend = os.path.splitext(path_name)[1]
    if (path_nameend.lower() == '.tif'):
        img_path.append(path_name)
# 开始读取矢量以及对应的栅格数据进行处理

for i in range(len(img_path)):
    img_file = ImgOpen(join(path_root, img_path[i]))
    out_name = img_path[i].split('.')[0] + 'shpcut.tif'
    out_root = join(save_root, out_name)
    # r = shapefile.Reader(shp_root)
    ds = gdal.Warp(out_root, # 裁剪图像保存完整路径（包括文件名）
                   img_file, # 待裁剪的影像
                   format='GTiff', # 保存图像的格式
                   cutlineDSName=shp_root, # 矢量文件的完整路径
                   cropToCutline=True, # 保证裁剪后影像大小跟矢量文件的图框大小一致（设置为False时，结果图像大小会跟待裁剪影像大小一样，则会出现大量的空值区域）
                   # cutlineWhere="FIELD = 'whatever'",
                   dstNodata=0)

