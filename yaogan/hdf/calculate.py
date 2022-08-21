import os
from PIL import Image
from matplotlib import pyplot as plt
from osgeo import gdal

from hdf.read_hdf_file import get_data


def getPAR(hdf_path,tiff_path):
    """ 根据提供的tiff范围获取hdf文件的par矩阵
        不能有中文路径
    """
    return get_data(hdf_path,tiff_path)

def read_tiff(path):
    """返回numpy"""
    ds = gdal.Open(path)
    im_width = ds.RasterXSize
    im_height = ds.RasterYSize
    im_data = ds.ReadAsArray(0, 0, im_width, im_height)
    return im_data

def calFPAR(b4,b5):
    arr4 = read_tiff(b4)
    arr5 = read_tiff(b5)
    ndvi = arr5 - arr4 / (arr5 + arr4)
    fpar = 1.24 * (ndvi) - 0.168
    return fpar

def calAPAR(fpar,par):
    return fpar * par

def calPRI(b2,b3):
    arr2 = read_tiff(b2)
    arr3 = read_tiff(b3)
    pri = ((0.53 * (arr3 - arr2)/(arr3 + arr2)) + 1)/2
    return pri

def calNPP(folder_path):
    files = os.listdir(folder_path)
    file_path = []
    for file in files:
        file_path.append(os.path.join(folder_path, file))
    b2 = file_path[0]
    b3 = file_path[1]
    b4 = file_path[2]
    b5 = file_path[3]
    par_path = file_path[4]

    pri = calPRI(b2,b3)

    fpar = calFPAR(b4,b5)
    par = getPAR(par_path,b2)
    apar = calAPAR(fpar,par)

    npp = 0.5139 * (pri * apar) - 1.9818
    return npp

if __name__ == '__main__':
    path = r"C:\Users\owl\Desktop\xiangmu"
    NPP_path = []
    folders = []
    for a,b,c in os.walk(path):
        for f in b:
            folders.append(os.path.join(a,f))
    dic = {0:"春分",1:"夏至",2:"秋分",3:"冬至"}
    # plt.imshow(npp, cmap='jet',vmax= 0.13381e+09)
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False  ## 设置正常显示符号
    for i in range(len(folders)):
        npp = calNPP(folders[i])
        plt.imshow(npp, cmap='jet', vmax=0.14381e+09)
        plt.colorbar()
        plt.title(dic[i], y=-0.2)
        fig = plt.gcf()
        fig.savefig(r'C:\Users\owl\Desktop\19年{}.png'.format(dic[i]), dpi=500, bbox_inches='tight')  # dpi越高越清晰
        plt.clf()
        plt.cla()
