import os
import shutil

import numpy as np
import h5py
from osgeo import gdal
from pyhdf.SD import SD
import matplotlib.pyplot as plt
from yaogan.coordinate import geo2lonlat, imagexy2geo
# os.environ['PROJ_LIB'] = r'C:\Users\owl\anaconda3\envs\pythonProject\Library\share\proj'
gdal.AllRegister()
"""查看hdf文件"""
def pic(path):
    hdf = SD(path)
    print(hdf.info())  # 信息类别数
    data = hdf.datasets()
    for i in data:
        print(i)  # 具体类别
        img = hdf.select(i)[:]  # 图像数据
        plt.imshow(img, cmap='gray')  # 显示图像
        plt.show()
def hdf5read(path):
    # HDF5的读取：
    with h5py.File(path, 'r') as f:  # 打开h5文件
        # 可以查看所有的主键
        for key in f.keys():
            print(f[key].name)
            print(f[key].shape)
            print(f[key].value)

"""获取tiff的经纬度范围，左上右下"""
def tiff_lurb(tiff_path):
    # 获取图像宽高以及转换六参数

    dataset = gdal.Open(tiff_path)
    print(dataset)
    # print(dataset.GetProjection())
    # getGeoTransform = dataset.GetGeoTransform()
    """影像左上角横坐标：geoTransform[0]
    影像左上角纵坐标：geoTransform[3]
    
    遥感图像的水平空间分辨率为geoTransform[1]
    遥感图像的垂直空间分辨率为geoTransform[5],为负数

    如果遥感影像方向没有发生旋转，即上北、下南，则
    geoTransform[2] 与 row *geoTransform[4] 为零。"""

    col = dataset.RasterXSize   #列数
    row = dataset.RasterYSize   #行数
    lu_geo = imagexy2geo(dataset, 0, 0)
    rb_geo = imagexy2geo(dataset, row, col)
    lu_jingwei = geo2lonlat(dataset,lu_geo[0],lu_geo[1])    # 左上经纬度
    rb_jingwei = geo2lonlat(dataset,rb_geo[0],rb_geo[1])    # 右下经纬度
    return [lu_jingwei[1],lu_jingwei[0],rb_jingwei[1],rb_jingwei[0]],[row,col]

"""获取tiff覆盖区域的PAR矩阵数据"""
def get_data(path,tiff):
    hdf = SD(path)
    #print(hdf.info())  # 信息类别数
    data = hdf.datasets()
    par_arry = []
    """读取hdf的矩阵数组"""
    for i in data:
        # print(i)  # 具体类别
        a = hdf.select(i)[:]
        par_arry = np.asarray(a)
    """读取tiff的经纬度范围和形状"""
    jingwei,tiff_shape = tiff_lurb(tiff)
    """匹配"""
    sh = par_arry.shape
    l,r = (jingwei[0]+180)/360 * sh[1], (jingwei[2]+180)/360 * sh[1]
    u,b = (90-jingwei[1])/180 * sh[0],(90-jingwei[3])/180 * sh[0]
    tiff_par_arry = par_arry[int(u):int(b)+2,int(l):int(r)+2]   #按经纬度比例取整

    new_arry = np.zeros(tiff_shape)
    row_1 = new_arry.shape[0]*30/5000
    col_1 = new_arry.shape[1]*30/5000

    for i in range(int(row_1) + 1 if isinstance(row_1,float) else row_1):   #如果结果是小数，则取整+1
        for j in range(int(col_1) + 1 if isinstance(col_1,float) else col_1):
            new_arry[int(i*5000/30):int((i+1)*5000/30),int(j*5000/30):int((j+1)*5000/30)] = tiff_par_arry[i][j]
    plt.imshow(new_arry, cmap='coolwarm')
    plt.colorbar()
    plt.show()

    return new_arry




"""读取hdf元数据信息"""
def pri_metadata(path):
    #  gdal打开hdf数据集
    datasets = gdal.Open(path)

    #  获取hdf中的子数据集
    SubDatasets = datasets.GetSubDatasets()
    #  获取子数据集的个数
    SubDatasetsNum = len(datasets.GetSubDatasets())
    #  输出各子数据集的信息
    print("子数据集一共有{0}个: ".format(SubDatasetsNum))
    for i in range(SubDatasetsNum):
        print(SubDatasets[i])

    #  获取hdf中的元数据
    Metadata = datasets.GetMetadata()
    #  获取元数据的个数
    MetadataNum = len(Metadata)
    #  输出各子数据集的信息
    print("元数据一共有{0}个: ".format(MetadataNum))
    for key, value in Metadata.items():
        print('{key}:{value}'.format(key=key, value=value))

def cantopen(root):
    """排查不能正常打开的"""
    cant = []
    total = 0
    for i in os.listdir(root):
        if os.path.isdir(os.path.join(root,i)):
            for par in os.listdir(os.path.join(root,i)):
                if par[-3:] == "hdf":
                    total += 1
                    # print(os.path.join(root,i,par))
                    try:
                        SD(os.path.join(root,i,par))
                    except:
                        cant.append(par.split(".")[2])

    print("total ",total)
    print("cant length ",len(cant))
    print(set(cant))

def move(root):

    for year in os.listdir(root):
        if os.path.isdir(os.path.join(root, year)):
            for par in os.listdir(os.path.join(root, year)):
                if par[-3:] == "hdf":
                    if not os.path.exists(os.path.join(root, "par", year)):
                        os.mkdir(os.path.join(root, "par", year))
                    shutil.copy(os.path.join(root, year, par), os.path.join(root, "par", year, par))

if __name__ == '__main__':
    root = r"F:\file\PAR"
    # cantopen(root)
    # move(root)
    path = r'F:\file\PAR\2016\GLASS04B01.V42.A2016109.2019363.hdf'
    tiff = r'F:\file\croped_europecity_5cloud\beijing\20130731\B5.TIF'
    # pic(path)
    get_data(path,tiff)
    # print(tiff_lurb(tiff))

