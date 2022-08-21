import os
import time
import torchvision
import numpy as np
from coordinate import lonlat2imagexy as tran_fn
from osgeo import gdal_array, gdal
from PIL import Image
from skimage import io


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
def get_image0(bands):
    '''融合波段，并返回融合后的数组结构'''
    pic_number = 0      #第1个波段目录
    B3_bath = bands[pic_number][0]
    B4_bath = bands[pic_number][1]
    B5_bath = bands[pic_number][2]
    '''读取不同波段'''
    B3 = gdal_array.LoadFile(B3_bath)
    B4 = gdal_array.LoadFile(B4_bath)
    B5 = gdal_array.LoadFile(B5_bath)
    img = np.stack([ B5, B4, B3])
    return img

def crop_img(dataset,img,jingwei):
    """
    根据读取的dataset裁剪图像并保存成假彩色png
    输入为array形式的数组
    """
    height = img.shape[1]
    width = img.shape[2]
    print('图像尺寸：',[height,width])
    jing1, wei1 = jingwei[0], jingwei[1]  # jingwei是两个点的坐标列表
    jing2, wei2 = jingwei[2], jingwei[3]
    hang_1, lie_1 = tran_fn(dataset, jing1, wei1)
    hang_2, lie_2 = tran_fn(dataset, jing2, wei2)
    print('裁剪后大小：',hang_2 - hang_1,lie_2 - lie_1)

    cropped = img[:,  # 通道不裁剪
              hang_1: hang_2,lie_1: lie_2]#用之前经纬度换算得到的行列数来裁剪
    t = time.localtime()
    #target = r'C:\Users\owl\Desktop\testtif' + '/cropped{}_{}{}{}{}.tif'.format("543",t.tm_mon, t.tm_mday,t.tm_hour, t.tm_min)
    #gdal_array.SaveArray(cropped, target, format="GTiff")
    """
    读取裁剪好的tif图片保存为png
    """

    """img_tif = io.imread(target)#这里读取出来直接就是[h,w,c]numpy数组
    img_8 = img_tif / 65535 * 255   #tif（unit16）转unit8
    out = Image.fromarray(img_8.astype('uint8'))
    out.save('C:\\Users\owl\Desktop\\testtif\\unit8{}_{}{}{}{}.png'.format("123",t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min))
    out.show()"""

if __name__ == '__main__':
    base_path = r'F:\文件\europe\paris_put'
    bands = read_landsat8_bands(base_path)
    img= get_image0(bands)
    dataset = gdal.Open(r"F:\文件\europe\paris_put\20170119\LC08_L1TP_199026_20170119_20200905_02_T1_B5.tif")  #这里的路径替换成bands
    jingwei = [2.21,48.9,2.47,48.8]
    crop_img(dataset,img,jingwei)
    print("波段列表为：{band}".format(band = bands))
    print("影像数量为：{imgs}".format(imgs = len(bands)))

