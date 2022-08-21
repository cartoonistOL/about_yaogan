from osgeo import gdal
import numpy as np
import os
from glob import glob
from math import ceil
import time
from coordinate import lonlat2imagexy as tran_fn

def crop_band(des_path,dataset,jingwei):

    jing1, wei1 = jingwei[0], jingwei[1]  # jingwei是两个点的坐标列表
    jing2, wei2 = jingwei[2], jingwei[3]
    hang_1, lie_1 = tran_fn(dataset, jing1, wei1)
    hang_2, lie_2 = tran_fn(dataset, jing2, wei2)
    print(hang_1,hang_2)
    print(lie_1,lie_2)
    src: gdal.Dataset = dataset
    target = des_path
    gdal.Translate(target, src, srcWin=[ lie_1,hang_1, lie_2-lie_1, hang_2-hang_1],
                   options=['-a_scale', '1'])

def GetExtent(infile):
    ds = gdal.Open(infile)
    geotrans = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    min_x, max_y = geotrans[0], geotrans[3]
    max_x, min_y = geotrans[0] + xsize * geotrans[1], geotrans[3] + ysize * geotrans[5]
    ds = None
    return min_x, max_y, max_x, min_y


def RasterMosaic(file_list, outpath,jingwei):
    Open = gdal.Open
    min_x, max_y, max_x, min_y = GetExtent(file_list[0])    #GetExtent获得四个角坐标

    for infile in file_list:
        minx, maxy, maxx, miny = GetExtent(infile)
        min_x, min_y = min(min_x, minx), min(min_y, miny)   #找到范围最大的四个点
        max_x, max_y = max(max_x, maxx), max(max_y, maxy)

    in_ds = Open(file_list[0])
    in_band = in_ds.GetRasterBand(1)
    geotrans = list(in_ds.GetGeoTransform())
    width, height = geotrans[1], geotrans[5]
    columns = ceil((max_x - min_x) / width)  # 列数
    rows = ceil((max_y - min_y) / (-height))  # 行数
    w,h = in_ds.RasterXSize,in_ds.RasterYSize
    outfile = outpath + "\\" + 'out12.tif'  # 结果文件名，可自行修改
    driver = gdal.GetDriverByName('GTiff')  # 新建一个图像
    out_ds = driver.Create(outfile, columns, rows, 1, in_band.DataType)
    out_ds.SetProjection(in_ds.GetProjection()) #复制投影坐标系
    geotrans[0] = min_x  # 更正左上角坐标
    geotrans[3] = max_y
    out_ds.SetGeoTransform(geotrans)
    out_band = out_ds.GetRasterBand(1)  #拿出第一个波段来写
    inv_geotrans = gdal.InvGeoTransform(geotrans)
    print("combining {} and {}......".format(file_list[0],file_list[1]))
    #轮流将影像写入新建的图片中
    old_data = []
    data = []
    old_x = 0
    for in_fn in file_list:
        in_ds = Open(in_fn)
        in_gt = in_ds.GetGeoTransform()
        offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
        x, y = map(int, offset)
        data = in_ds.GetRasterBand(1).ReadAsArray()
        if len(old_data) == 0:
            old_data = data
            old_x = x
        #两张图片需要在重叠部分做一些数值处理，例如先来后到、取最大值等
        else:
            # a1、a2: 两张图重叠部分的array
            # h、w：高、宽
            # old_x：横向偏移列数
            # y：纵向偏移行数
            a1 = old_data[y:h,0:w-old_x]
            a2 = data[0:h-y,old_x:w]

            for i in range(h-y):
                for j in range(w-old_x):
                    if a1[i][j] != 0:
                        #先来后到
                        a2[i][j] = a1[i][j]
            data[0:h-y,old_x:w] = a2
    #拼接结束后进行裁剪
    #out_ds.GetRasterBand(1).WriteArray(data)
    #crop_band(outfile,out_ds,jingwei)
    #data = np.where(data,data,np.NAN)       #把0值全部置为nan
            out_band.WriteArray(data, x, y)  # x，y是开始写入时左上角像元行列号
        #crop_band(outfile, outfile, jingwei)

    del in_ds, out_band, out_ds
    return outfile


def compress(path, target_path, method="LZW"):  #
    """使用gdal进行文件压缩，
          LZW方法属于无损压缩，
          效果非常给力，4G大小的数据压缩后只有三十多M"""
    dataset = gdal.Open(path)
    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(target_path, dataset, strict=1, options=["TILED=YES", "COMPRESS={0}".format(method)])
    del dataset


if __name__ == '__main__':
    path = r'C:\Users\owl\Desktop\test\t2'  # 该文件夹下存放了待拼接的栅格
    outpath = r'C:\Users\owl\Desktop\test\out'
    os.chdir(path)
    raster_list = sorted(glob('*.tif'))  # 读取文件夹下所有tif数据
    jingwei = [120.7543,31.9024,122.0582,30.6608]
    result = RasterMosaic(raster_list,outpath, jingwei)  # 拼接栅格
    #compress(result, target_path=r'J:\backup\Global.tif')  # 压缩栅格