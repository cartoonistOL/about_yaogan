import os
from PIL import Image
import numpy as np
from osgeo import gdal
from osgeo import osr
# os.environ['PROJ_LIB'] = r'C:\Users\owl\anaconda3\envs\pythonProject\Library\share\proj'
gdal.AllRegister()
def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs
def lonlat2geo(dataset, lon, lat):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lat, lon)
    return coords[:2]
def geo2imagexy(dataset, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    '''print("trans各值为",trans)'''
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    '''print("a的值为",a)
    print("b的值为",b)
    print("方程解为：",np.linalg.solve(a, b))'''
    A,B = np.linalg.solve(a, b)
    return B,A   # 使用numpy的linalg.solve进行二元一次方程的求解

def imagexy2geo(dataset, row, col):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    trans = dataset.GetGeoTransform()
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return px, py

def geo2lonlat(dataset, x, y):
    '''
    将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据肯定）
    :param dataset: GDAL地理数据
    :param x: 投影坐标x
    :param y: 投影坐标y
    :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(x, y)
    return coords[:2]

def lonlat2imagexy(dataset,x, y):
    '''
    经纬度转影像行列：
    此函数经纬度位置是相反的
    ：通过经纬度转平面坐标
    ：平面坐标转影像行列
    '''
    coords = lonlat2geo(dataset, x, y)
    coords2 = geo2imagexy(dataset,coords[0], coords[1])
    return (int(round(abs(coords2[0]))), int(round(abs(coords2[1]))))



if __name__ == '__main__':
    dataset = gdal.Open(r"F:\file\croped_europecity_5cloud\beijing\20130901\B2.tif")

    adfGeoTransform = dataset.GetGeoTransform()  # 读取地理信息
    arrSlope = []  # 用于存储每个像素的（X，Y）坐标
    nXSize = dataset.RasterXSize  # 列数
    nYSize = dataset.RasterYSize  # 行数
    for i in range(nYSize):
        row = []
        for j in range(nXSize):
            px = adfGeoTransform[0] + i * adfGeoTransform[1] + j * adfGeoTransform[2]
            py = adfGeoTransform[3] + i * adfGeoTransform[4] + j * adfGeoTransform[5]
            lat,lon = geo2lonlat(dataset,px,py)
            col = [lon,lat]
            row.append(col)
        arrSlope.append(row)

    print(len(arrSlope))


    # dataset = gdal.Open(r"F:\文件\crop_2345band\changsha\20131012\merged_B2.tif")
    # x, y = 111.890861,28.664368
    #
    # """ result : 3263 1448
    #     envi : 1448.2686, 3263.2442"""
    # x4,y4 = 114.256514,27.851024
    # """ result :7911 5520
    #     envi :5519.5475,7910.8591
    # """
    # xoffset, yoffset = lonlat2imagexy(dataset, x, y)
    # print(xoffset,yoffset)
    #
    # xoffset4, yoffset4 = lonlat2imagexy(dataset, x4, y4)
    # print(xoffset4,yoffset4)
    # #print(Image.open(r'C:\Users\owl\Desktop\testtif\LC08_L1TP_123032_20170115_20200905_02_T1_refl.TIF').size)
    # #切割图片
    # src: gdal.Dataset = dataset
    # src = gdal.Translate(r'C:\Users\owl\Desktop\tt4.tif', src, srcWin=[yoffset, xoffset, yoffset4-yoffset, xoffset4-xoffset],
    #                    options=['-a_scale', '1'])
    """print("切割后的图片尺寸为({},{})".format(xoffset4-xoffset,yoffset4-yoffset))
    print('坐标转换-对应行列像素位置')
    print('(%s, %s)->(%s, %s)' % (x, y, xoffset, yoffset))
    print('(%s, %s)->(%s, %s)' % (x4, y4, xoffset4, yoffset4))
    print('----------------------------------------------------------------------------------------------')
    print('数据投影：')
    projection = dataset.GetProjection()
    print(projection)

    print('数据的大小（行，列）：')
    print('(%s %s)' % (dataset.RasterYSize, dataset.RasterXSize))

    geotransform = dataset.GetGeoTransform()
    print('地理坐标：')
    print(geotransform)"""

