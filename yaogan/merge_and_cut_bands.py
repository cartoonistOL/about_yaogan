"""
    合并、裁剪Landsat8影像
"""

import os
import sys
import time
import traceback

import osgeo_utils.gdal_merge as gm
from osgeo import gdal

from yaogan.coordinate import lonlat2imagexy
from coordinate import lonlat2geo, geo2imagexy

"""主方法，按经纬度裁剪图像并输出
    des_path：输出文件路径.tif
    band_path：输入文件路径.tif
    jingwei：左上、右下的经纬度
"""


def crop_band(des_path, band_path, jingwei):  # 方法2
    dataset = gdal.Open(band_path)
    jing1, wei1 = jingwei[0], jingwei[1]  # jingwei是两个点的坐标列表
    jing2, wei2 = jingwei[2], jingwei[3]
    hang_1, lie_1 = lonlat2imagexy(dataset, jing1, wei1)
    hang_2, lie_2 = lonlat2imagexy(dataset, jing2, wei2)
    print(hang_1, lie_1)
    print(hang_2, lie_2)
    src: gdal.Dataset = dataset
    target = des_path
    gdal.Translate(target, src, srcWin=[lie_1, hang_1, lie_2 - lie_1, hang_2 - hang_1],
                   options=['-a_scale', '1'])
    if lie_2 - lie_1 > 0 and hang_2 - hang_1 > 0:
        may_can_use.append(des_path)  # 如果成功裁剪，计入可用影像列表
        print('{}已保存'.format(target))


"""只裁剪，不合并的方法"""


def cut(date_path, out_path, band_file):
    os.chdir(date_path)
    band = band_file.split("_")[-1][:2]  # file1 : LC08_L1TP_122040_20130724_20200912_02_T1_B2.tif，merged_B2.tif
    city_date = date_path.split("\\")[-2] + "\\" + date_path.split("\\")[
        -1]  # date_path :F:\文件\landsat8\changsha\20130724
    out_file_path = out_path + "\\" + city_date
    if not os.path.exists(out_file_path):  # 不存在目录则创建
        os.makedirs(out_file_path)
    target_croped_path = out_file_path + "\\" + "croped_{}.TIF".format(band)
    if not os.path.exists(target_croped_path):
        print(band_file, "is cutting...")
        jingwei = coor["{}".format(date_path.split("\\")[-2])]  # 经纬度范围的左,上,右,下
        crop_band(target_croped_path, band_file, jingwei)


""" 合并并裁剪
    date_path：日期目录路径
    out_path :与根目录结构相同
    file_list：待处理波段文件，数量大于1"""


def merge_and_cut(date_path, out_path, file_list):
    os.chdir(date_path)
    band = file_list[0].split("_")[-1][:2]  # file1 : LC08_L1TP_122040_20130724_20200912_02_T1_B2.tif
    city_date = date_path.split("\\")[-2] + "\\" + date_path.split("\\")[
        -1]  # date_path :F:\文件\landsat8\changsha\20130724
    out_file_path = out_path + "\\" + city_date
    if not os.path.exists(out_file_path):  # 不存在目录则创建
        os.makedirs(out_file_path)

    target_merged_path = out_file_path + "\\" + "merged_{}.TIF".format(band)
    target_croped_path = out_file_path + "\\" + "croped_{}.TIF".format(band)
    if not os.path.exists(target_croped_path):  # 重复文件就不生成了
        args = r"-o {} -n 0 ".format(target_merged_path) + " ".join(i for i in file_list)
        # coor = ["-ul_lr"] + jingwei.split(",")
        print(file_list, "is merging...")
        gm.main([" "] + args.split(" "))  # args输入到命令行
        print(target_merged_path, "is cutting...")
        jingwei = coor["{}".format(date_path.split("\\")[-2])]  # 经纬度范围的左,上,右,下
        print("jinwei : ", jingwei)
        crop_band(target_croped_path, target_merged_path, jingwei)
        os.remove(target_merged_path)  # 合成后删除已拼接文件


"""可用日期统计"""


def cal(out_path):
    citys = os.listdir(out_path)
    ls = []
    for city in citys:
        city_path = out_path + "\\" + city
        dates = os.listdir(city_path)
        for date in dates:
            date_path = city_path + "\\" + date
            files = os.listdir(date_path)
            for file in files:
                if "croped" in file:
                    ls.append(date_path)
    print(len(list(set(ls))))


"""全局坐标
    guangzhou :122044 shenyang:119030,119031 zhenzhou:124036 
    """
coor = {"guangzhou": [112.8886, 23.9022, 114.1367, 22.5304], "shanghai": [120.9399, 31.8951, 122.0582, 30.6608],
        "nanjing": [118.3117, 32.6385, 119.3290, 31.1913], "hefei": [116.5415, 32.6002, 118.0508, 30.9071],
        "xian": [107.7988, 34.6837, 109.3292, 33.7274], "chengdu": [102.989623, 31.437765, 104.896262, 30.090979],
        "changsha": [112.4401, 28.6812, 114.1639, 27.8472], "shenyang": [122.5443, 42.9403, 123.8084, 41.1979],
        "zhenzhou": [112.714711, 34.984219, 114.241034, 34.264796]
        }


def count(out_path):
    citys = os.listdir(out_path)
    ls = []
    city_list = []
    for city in citys:
        print(city, "is working...")
        city_path = out_path + "\\" + city
        dates = os.listdir(city_path)
        for date in dates:
            date_path = city_path + "\\" + date
            files = os.listdir(date_path)
            for file in files:
                if "croped" in file:
                    ls.append(date_path)
                    city_list.append(city)
    print(list(set(city_list)))
    print(len(list(set(ls))))


may_can_use = []


def run():
    """changsha:, shenyang:, shanghai:，chengdu(需要三张)"""
    error_num_date = []
    unfit_date = []
    time0 = time.time()
    root_path = r'F:\文件\landsat8'  # 包含各个城市的文件夹
    out_path = r'F:\文件\crop_2345band'
    dont_merge_list = ['guangzhou', 'zhenzhou']  # 如果不需要合并的城市不在列表中，则会被跳过
    citys = os.listdir(root_path)

    for city in citys:
        if city in dont_merge_list:  # 如果是覆盖的城市，就不需要合并
            dont_merge = 1
        else:
            dont_merge = 0
        print(city, "is working...")
        time1 = time.time()
        city_path = root_path + "\\" + city
        dates = os.listdir(city_path)
        for date in dates:
            date_path = city_path + "\\" + date
            files = os.listdir(date_path)
            if dont_merge:
                """直接裁剪"""
                for i in files:
                    cut(date_path, out_path, i)
            elif len(files) > 4:  # 可能有重复波段的文件夹，重复则说明有可能需要合并
                b2 = [band for band in files if 'B2' in band]
                b3 = [band for band in files if 'B3' in band]
                b4 = [band for band in files if 'B4' in band]
                b5 = [band for band in files if 'B5' in band]
                number = min([len(b2), len(b3), len(b4), len(b5)])

                if city == 'chengdu' and number < 3:
                    continue
                """如果有重叠部分"""
                if number > 1:
                    for bands in b2, b3, b4, b5:
                        try:
                            if len(b2) + len(b3) + len(b4) + len(b5) != len(b2) * 4:  # 如果各波段数量之间有不一致的，最后也用不了
                                unfit_date.append(dates)
                            else:
                                merge_and_cut(date_path, out_path, bands)
                        except (Exception, BaseException) as e:
                            print('{:*^60}'.format('使用traceback的format_exc可以输出错误具体位置'))
                            exstr = traceback.format_exc()
                            print(exstr)
                else:
                    unfit_date.append(date_path)
            elif len(files) < 4:
                # print("{} 文件数量不足4".format(date_path))
                error_num_date.append(date_path)
        print(city, "is done,cost ", time.time() - time1)
    print("-------------done---------------")
    print("影像数量不足4列表：", error_num_date)
    print("处理影像共{}幅：\n ".format(len(may_can_use)), may_can_use)
    print("可用日期数：", end=" ")
    cal(out_path)
    print("total cost {}".format(time.time() - time0))
    # file_path = r'C:\Users\owl\Desktop\test\t2'


if __name__ == '__main__':
    count(r'F:\文件\crop_2345band')  # 结构为 城市\\日期\\各波段文件.tif，当文件数量超过4，说明这一天有重叠影像，可能需要合并切割

