import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import datetime

import pandas as pd

start = datetime.datetime.now()
from skimage import io
import seaborn as sns


split_range = (0, 256 * 256)
split_num = 256

def city_rename(city_path):
    for date in os.listdir(city_path):
        date_path = os.path.join(city_path,date)
        file_path = os.path.join(date_path,os.listdir(date_path)[0])
        os.rename(file_path,date_path + '\\{}_{}_543.tif'.format(city_path.split("\\")[-1],date))

def get_fenbu(path):
    img = io.imread(path)
    r = img[:, :, 0].reshape(-1)
    g = img[:, :, 1].reshape(-1)
    b = img[:, :, 2].reshape(-1)
    r = np.histogram(r,split_num,split_range)[0]   #np.histogram构建频数分布图
    g = np.histogram(g,split_num,split_range)[0]     #返回的二元组，该函数返回一个二元组(f, b)，其中f为含有m个整数的数组，每个整数表示对应区间频数。b代表区间端值
    b = np.histogram(b,split_num,split_range)[0]
    #print(r)
    #是否取均值
    #r = (r+g+b) / 3
    return r,g,b

def get_KL_from_fenbu(x,y):
    """比较两个频数分布的KL散度
    """
    sumx = np.sum(x)
    sumy = np.sum(y)
    print(x)
    print(y)
    for i in range(len(x)):
        x[i] = x[i] / sumx
    for i in range(len(y)):
        y[i] = y[i] / sumy
    print(x)
    print(y)
    """x = torch.tensor(x)
    y = torch.tensor(y)
    logp_x = F.log_softmax(x, dim=-1)
    p_y = F.softmax(y, dim=-1)
    KL = F.kl_div(logp_x, p_y, reduction='sum')"""
    """for i in range(split_num):
        px = x[i] / sumx
        py = y[i] / sumy
        if y[i] == 0 or x[i] == 0:
            KL += px
        else:
            KL += px * math.log2(px / py)"""
    return KL

def path2KL(path_base,path_1,channel):
    """ 输入两个需要比较的543图片路径
        返回通道的KL散度
        """
    fenbu_base_r, fenbu_base_g, fenbu_base_b = get_fenbu(path_base)
    #print(fenbu_base_r,fenbu_base_g,fenbu_base_b)
    fenbu_1_r, fenbu_1_g, fenbu_1_b = get_fenbu(path_1)
    #print(fenbu_1_r, fenbu_1_g, fenbu_1_b)
    '''KL_baseto1_r = scipy.stats.entropy(fenbu_base_r,fenbu_1_r)      #scipy.stats.entropy计算两个分布的KL散度
    KL_baseto1_g = scipy.stats.entropy(fenbu_base_g, fenbu_1_g)
    KL_baseto1_b = scipy.stats.entropy(fenbu_base_b, fenbu_1_b)'''
    if channel == 'r':
        KL_baseto1_r = get_KL_from_fenbu(fenbu_base_r, fenbu_1_r)
        return KL_baseto1_r
    elif channel == 'g':
        KL_baseto1_g = get_KL_from_fenbu(fenbu_base_g, fenbu_1_g)
        return KL_baseto1_g
    elif channel == 'b':
        KL_baseto1_b = get_KL_from_fenbu(fenbu_base_b, fenbu_1_b)
        return KL_baseto1_b
    else:
        print("通道错误")

def jijie_split(city_path,lichun,lixia,liqiu,lidong):
    """获取四季图片的路径"""
    spring = []
    summer = []
    autumn = []
    winter = []

    for date in os.listdir(city_path):
        date_path = city_path + "\\" + date
        file_path = date_path + "\\" + os.listdir(date_path)[0]
        day = (int)(date[4:8])
        # print(day)
        if day >= lichun and day < lixia:
            spring.append(file_path)
        elif day >= lixia and day < liqiu:
            summer.append(file_path)
        elif day >= liqiu and day < lidong:
            autumn.append(file_path)
        elif day >= lidong or day < lichun:
            winter.append(file_path)
    arr = [spring,summer,autumn,winter]
    return arr

""" 二维KL散度热力图，分组多对多的比较,春-秋，夏-冬
    输入该城市某一年的文件路径
    格式为city_path/date/543.tiff
    """
def get_juzhen_from_jijie(jijie1,jijie2,channel):
    """ 以两季列表作为输入，获得指定通道的kl散度矩阵"""
    arr = np.zeros((len(jijie2),len(jijie1)),dtype=float)   #春作列，其他季节作为行
    for i in range(len(jijie1)):
        for j in range(len(jijie2)):
            kl_r = path2KL(jijie1[i],jijie2[j],'r')
            kl_g = path2KL(jijie1[i],jijie2[j],'g')
            kl_b = path2KL(jijie1[i], jijie2[j], 'b')
            arr[j][i] = (kl_r + kl_g + kl_b) / 3
    return arr

def relitu(siji,target,jijie1num,jijie2num,index,way):
    
    r_kl_juzhen_chunqiu = get_juzhen_from_jijie(siji[jijie1num], siji[jijie2num], 'r')
    data_1 = {}
    for i in range(len(index[jijie1num])):
        data_1[index[jijie1num][i]] = r_kl_juzhen_chunqiu[:, i]
    pd_data = pd.DataFrame(data_1, index=index[jijie2num], columns=index[jijie1num])
    ax = sns.heatmap(pd_data, annot=True, fmt='.3f', vmin=0.0, vmax=5.0)
    plt.xlabel('spring', fontsize=14, color='k')  # x轴label的文本和字体大小
    plt.ylabel('autumn', fontsize=14, color='k')  # y轴label的文本和字体大小
    scatter_fig = ax.get_figure()
    scatter_fig.savefig(target + "\\" + way + ".png", dpi=400)
    plt.close()

def KL_multi(city_path):
    siji = jijie_split(city_path,204,506,808,1108)

    """获取绘图坐标"""
    spring,summer,autumn,winter = [],[],[],[]
    index = [spring,summer,autumn,winter]
    for i in range(len(siji)):
        for j in siji[i]:
            index[i].append((j.split("\\")[-2])[2:])

    """ 获得季节多对多kl散度矩阵
        并绘制热力图
        春-秋，夏-冬"""
    target = r'C:\Users\owl\Desktop'

    relitu(siji, target, 0, 2, index,'春秋')
    relitu(siji, target, 0, 1, index,'春夏')
    relitu(siji, target, 1, 2, index,'夏秋')
    relitu(siji, target, 1, 3, index,'夏冬')

""" 四季kl均值比较
    输入该城市某一年的文件路径
    格式为city_path/date/543.tiff
    """
def KL_avg(city_path):
    siji = jijie_split(city_path, 204, 506, 808, 1108)
    """获取四季的分布的平均值"""
    fenbu_r_avg = []
    fenbu_g_avg = []
    fenbu_b_avg = []
    for jijie in siji:
        arr_r = [0] * split_num
        arr_g = [0] * split_num
        arr_b = [0] * split_num
        for day in jijie:
            r,g,b = get_fenbu(day)
            arr_r += r
            arr_g += g
            arr_b += b
        arr_r = np.asarray(arr_r)
        arr_g = np.asarray(arr_g)
        arr_b = np.asarray(arr_b)
        arr_r = arr_r / len(jijie)
        arr_g = arr_g / len(jijie)
        arr_b = arr_b / len(jijie)
        arr_r.tolist()
        arr_g.tolist()
        arr_b.tolist()
        fenbu_r_avg.append(arr_r)
        fenbu_g_avg.append(arr_g)
        fenbu_b_avg.append(arr_b)
    kl_chunqiu_r = get_KL_from_fenbu(fenbu_r_avg[0],fenbu_r_avg[2])
    kl_chunqiu_g = get_KL_from_fenbu(fenbu_g_avg[0], fenbu_g_avg[2])
    kl_chunqiu_b = get_KL_from_fenbu(fenbu_b_avg[0], fenbu_b_avg[2])
    print("春-秋平均值KL散度：{}".format((kl_chunqiu_r + kl_chunqiu_g + kl_chunqiu_b) / 3))
    kl_chunxia_r = get_KL_from_fenbu(fenbu_r_avg[0], fenbu_r_avg[1])
    kl_chunxia_g = get_KL_from_fenbu(fenbu_g_avg[0], fenbu_g_avg[1])
    kl_chunxia_b = get_KL_from_fenbu(fenbu_b_avg[0], fenbu_b_avg[1])
    print("春-夏平均值KL散度：{}".format((kl_chunxia_r + kl_chunxia_g + kl_chunxia_b) / 3))
    kl_xiadong_r = get_KL_from_fenbu(fenbu_r_avg[1], fenbu_r_avg[3])
    kl_xiadong_g = get_KL_from_fenbu(fenbu_g_avg[1], fenbu_g_avg[3])
    kl_xiadong_b = get_KL_from_fenbu(fenbu_b_avg[1], fenbu_b_avg[3])
    print("夏-冬平均值KL散度：{}".format((kl_xiadong_r + kl_xiadong_g + kl_xiadong_b) / 3))
    kl_xiaqiu_r = get_KL_from_fenbu(fenbu_r_avg[1], fenbu_r_avg[2])
    kl_xiaqiu_g = get_KL_from_fenbu(fenbu_g_avg[1], fenbu_g_avg[2])
    kl_xiaqiu_b = get_KL_from_fenbu(fenbu_b_avg[1], fenbu_b_avg[2])
    print("夏-秋平均值KL散度：{}".format((kl_xiaqiu_r + kl_xiaqiu_g + kl_xiaqiu_b) / 3))


if __name__ == "__main__":
    root = r'C:\Users\owl\Desktop\tifftest'
    city_path = r'C:\Users\owl\Desktop\tifftest\beijing_2019'
    #path2KL(r'C:\Users\owl\Desktop\tifftest\beijing_2019\20190206\combine_543.tif',r'C:\Users\owl\Desktop\tifftest\beijing_2019\20190310\combine_543.tif')
    #KL_multi(city_path)
    #a = np.random.rand(3,3)
    KL_avg(city_path)

    end = datetime.datetime.now()
    print("total time is {}".format(end - start))







