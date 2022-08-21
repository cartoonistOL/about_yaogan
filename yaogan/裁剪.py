import os
from pathlib import Path
from coordinate import lonlat2imagexy as tran_fn
from osgeo import gdal_array, gdal


def ComposeisRight(city_path):      #检查最终文件结构
    faul_count = 0
    faul_format = 0
    for file in os.listdir(city_path):
        filelist = os.listdir(os.path.join(city_path, file))
        date_path = os.path.join(city_path, file)
        if len(filelist) != 4:
            faul_count += 1
            print(date_path + '中数量不对，为{}'.format(len(filelist)))
        else:
            filelist = os.listdir(date_path)
            if not 'B2' in filelist[0] and 'B3' in filelist[1] and 'B4' in filelist[2] and 'B5' in filelist[3]:
                print("{}中文件格式不正确".format(date_path))
    print('{}中{}个问题'.format(city_path.split("\\")[-1],faul_count + faul_format))

def read_landsat8_bands(base_path):
    """保存landsat8不同波段的路径(共11个波段)
    base_path: 存储了多张影像的文件夹
    mid_path: 存储了不同波段的文件夹，对应着单张影像
    final_path: 最后一层可以直接打开的单张波段文件

    bands：包含了波段路径的列表
    """

    # 用于存储不同波段路径,维度是影像数量*波段数（11）
    bands = []
    num_bands = 11

    # 用于定位波段用的关键字列表
    keys = []
    for k in range(num_bands):
        KYE,key = 'B{num}.TIF'.format(num = k + 1), 'B{num}.tif'.format(num = k + 1)
        keys.append(KYE)
        keys.append(key)

    # 读取最外层文件
    base_files = os.listdir(base_path)
    for i in range(len(base_files)):
        bands.append([])

        # 读取中层文件
        mid_path = base_path + '\\' + base_files[i]
        mid_file = os.listdir(mid_path)

        # 得到最内层的波段文件
        for final_file in mid_file:
            final_path = mid_path + '\\' + final_file

            for j in range(num_bands):
                if keys[j] in final_file:
                    bands[i].append(final_path)

        # 原始列表排序是1,10,11,2,3，...
        # 按照倒数第5个字符进行排序（XXXB1.TIF）
        bands[i].sort(key=lambda arr: (arr[:-5], int(arr[-5])))

    # 返回波段列表和影像数量
    return bands

def crop_band1(des_path,band_path,jingwei):       #方法1
    """
    根据读取的dataset裁剪图像并保存成假彩色png
    输入为band的路径
    """
    dataset = gdal.Open(band_path)
    img = gdal_array.LoadFile(band_path)
    height = img.shape[0]
    width = img.shape[1]
    jing1, wei1 = jingwei[0], jingwei[1]  # jingwei是两个点的坐标列表
    jing2, wei2 = jingwei[2], jingwei[3]
    hang_1, lie_1 = tran_fn(dataset, jing1, wei1)
    hang_2, lie_2 = tran_fn(dataset, jing2, wei2)
    cropped = img[hang_1: hang_2,lie_1: lie_2]#用之前经纬度换算得到的行列数来裁剪
    target = des_path
    gdal_array.SaveArray(cropped, target, format="GTiff")
    print('{}已保存'.format(target))

def crop_band2(des_path,band_path,jingwei):       #方法2
    dataset = gdal.Open(band_path)
    jing1, wei1 = jingwei[0], jingwei[1]  # jingwei是两个点的坐标列表
    jing2, wei2 = jingwei[2], jingwei[3]
    hang_1, lie_1 = tran_fn(dataset, jing1, wei1)
    hang_2, lie_2 = tran_fn(dataset, jing2, wei2)
    src: gdal.Dataset = dataset
    target = des_path
    gdal.Translate(target, src, srcWin=[ lie_1,hang_1, lie_2-lie_1, hang_2-hang_1],
                   options=['-a_scale', '1'])
    print('{}已保存'.format(target))

def crop(root_path,des_path,dicOfcity):

    for city in dicOfcity:  # 遍历城市
        jingwei = dicOfcity[city]
        city_path = root_path + "\\" + city  # 城市文件夹路径

        print("裁剪前检查")
        ComposeisRight(city_path)
        print("__________________________________________________")
        bands = read_landsat8_bands(city_path)  # 该城市所有图片的波段信息 ，第一维是日期,第二维是波段路径
        des_city_path = os.path.join(des_path, city)

        for bands_axis2 in bands:  # 遍历日期，bands_axis2为单个日期文件夹中包含的波段文件路径列表
            date = bands_axis2[0].split('\\')[-2]
            des_date_path = os.path.join(des_city_path, date)
            if not os.path.isdir(des_date_path):  # 如果目标日期文件夹不存在
                os.makedirs(des_date_path)      #递归创建目录
            if len(os.listdir(des_date_path)) == 4:  # 如果目标日期文件夹数量为4
                print("{}裁剪图像已存在".format(des_date_path))
            else:
                for band in bands_axis2:
                    des_file_path = des_date_path + '\\' + band.split('_')[-1]
                    crop_band2(des_file_path,band,jingwei)      #方法二为官方方法
        print("__________________________________________________")
        print("裁剪后检查")
        ComposeisRight(des_city_path)
        print("__________________________________________________")

if __name__ == '__main__':
    root_path = r'F:\文件\europe'  # 根目录，该目录下包含各个城市文件夹
    des_path = r'F:\文件\croped_city'  # 目标目录，结构与根目录相同
    dicOfcity = {'vienna': [16.1640,48.3350,16.5939,48.1150]}
    #dicOfcity = {'beijing': [116.0630,40.1869,116.7300,39.6824], 'berlin': [13.0558, 52.68, 13.7780, 52.3375],
    #             'budapest': [18.9066, 47.6088, 19.3621, 47.3349],'london': [0.5910, 51.7323, 0.3318, 51.2425],
    #             'riga': [23.9088, 57.0973, 24.3437, 56.8499],'roma': [12.3211, 42.0582, 12.7417,41.7716],
    #             'vienna': [16.1640,48.3350,16.5939,48.1150],'bremen': [8.4727,53.2324,8.9956,53.0078],
    #             'dublin': [-6.4684,53.4317,-6.0260,53.2169]}
    crop(root_path, des_path, dicOfcity)
