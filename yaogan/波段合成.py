import os

import numpy as np
from osgeo import gdal
from format_test import ComposeisRight, CombineisRight


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



'''def combine_image_out(bands_axis,des_date_path,type):
        #bands_axis：单张影像的波段列表
        #des_date_path：最终目标目录
        #type：’543‘ or ’432‘
    B2_bath = bands_axis[0]
    B3_bath = bands_axis[1]
    B4_bath = bands_axis[2]
    B5_bath = bands_axis[3]
    #读取不同波段
    B2 = gdal_array.LoadFile(B2_bath)
    B3 = gdal_array.LoadFile(B3_bath)
    B4 = gdal_array.LoadFile(B4_bath)
    B5 = gdal_array.LoadFile(B5_bath)
    if type == '432':
        img = np.stack([B4, B3, B2])
        target = des_date_path + '/combine_{}.tif'.format('432')
        gdal_array.SaveArray(img, target, format="GTiff")
        print("{}合成成功".format(target))
    elif type == '543':
        img = np.stack([ B5, B4, B3])
        target = des_date_path + '/combine_{}.tif'.format('543')
        gdal_array.SaveArray(img, target, format="GTiff")
        print("{}合成成功".format(target))
        #读取裁剪好的tif图片保存为png

    """img_tif = io.imread(target)#这里读取出来直接就是[h,w,c]numpy数组
    img_8 = img_tif / 65535 * 255   #tif（unit16）转unit8
    out = Image.fromarray(img_8.astype('uint8'))
    out.save('C:\\Users\owl\Desktop\\testtif\\unit8{}_{}{}{}{}.png'.format("123",t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min))
    out.show()"""
    '''


class GRID:

    # 读图像文件
    def read_img(self, filepath):
        dataset = gdal.Open(filepath)  # 打开文件

        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数

        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_proj = dataset.GetProjection()  # 地图投影信息
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵

        del dataset  # 关闭对象，文件dataset
        return im_proj, im_geotrans, im_data, im_width, im_height

    # 写文件，以写成tif为例
    def write_img(self, filepath, im_proj, im_geotrans, im_data):

        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filepath, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

        del dataset


def combine_image(bands, des_date_path,type):
    file_path = des_date_path + '\\combine_{}.tif'.format(type)
    if not os.path.exists(file_path):
        run = GRID()
        # 第一步,文件结构为包含5，4，3，2波段的四个文件
        proj, geotrans, data2, row1, column1 = run.read_img(bands[0])  # 读数据
        proj, geotrans, data3, row2, column2 = run.read_img(bands[1])  # 读数据
        proj, geotrans, data4, row3, column3 = run.read_img(bands[2])  # 读数据
        proj, geotrans, data5, row4, column4 = run.read_img(bands[3])  # 读数据
        if type == "rgb":
            data432 = np.array((data4, data3, data2), dtype=data2.dtype)  # 按序将3个波段像元值放入
            run.write_img(file_path, proj, geotrans, data432)  # 写为3波段数据
            print('{}已保存'.format(des_date_path + '\\combine_432.tif'))
        elif type == "falseColor":
            data543 = np.array((data5, data4, data3), dtype=data2.dtype)  # 按序将3个波段像元值放入
            run.write_img(file_path, proj, geotrans, data543)  # 写为3波段数据
            print('{}已保存'.format(des_date_path + '\\combine_falseColor.tif'))
        else:
            print('格式错误，不存在{}，请选择rgb或falseColor'.format(type))
    else:
        print("{}已存在".format(file_path))

def combine(root_path,des_path,dicOfcity,combine_type):
    """combine_type：合成方式（rgb，falseColor）"""
    for city in dicOfcity:     #遍历城市
        print("=================starting {}=====================".format(city))
        city_path = root_path + "\\" + city    #城市文件夹路径
        if not os.path.isdir(city_path):
            continue
        des_city_path = os.path.join(des_path, city)
        bands = read_landsat8_bands(city_path)      #该城市所有图片的波段信息 ，第一维是日期,第二维是波段路径
        print("__________________________________________________")
        print("合成前检查")
        flag = ComposeisRight(city_path)
        print("__________________________________________________")
        if flag == 1:
            for bands in bands:  # 遍历日期，bands为单个日期文件夹中包含的波段文件路径列表
                date = bands[0].split('\\')[-2]
                des_date_path = os.path.join(des_city_path, date)
                if not os.path.isdir(des_date_path):  # 如果目标日期文件夹不存在
                    os.makedirs(des_date_path)
                if len(os.listdir(des_date_path)) == 2:  # 如果目标日期文件夹不为空
                    print("{}合成图像已存在".format(des_date_path))
                else:
                    combine_image(bands, des_date_path,combine_type)
            print("合成后检查")
            CombineisRight(des_city_path,1)
            print("__________________________________________________")



if __name__ == '__main__':
    root_path = r'F:\文件\crop_2345band'  # 根目录，该目录下包含各个城市文件夹
    des_path = r'F:\文件\new_cities_combined'  # 目标目录，结构与根目录相同
    #dicOfcity = {'beijing': [],'berlin': [],'bremen': [],'brussel': [],'budapest':[]}
    dicOfcity = {'guangzhou': [],'changsha': [],'hefei': [],'nanjing': [],'shanghai': [],'shenyang': [],'xian': [],'zhenzhou': []}
    combine(root_path, des_path, dicOfcity,"rgb")
    combine(root_path, des_path, dicOfcity, "falseColor")



