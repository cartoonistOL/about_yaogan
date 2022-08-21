import math
import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import datetime

import pandas as pd

start = datetime.datetime.now()
from skimage import io
import seaborn as sns

#分割范围
split_range = (0, 256 * 256)
#每个通道分割数量
split_num = 256



def city_rename(city_path):
    for date in os.listdir(city_path):
        date_path = os.path.join(city_path, date)
        file_path = os.path.join(date_path, os.listdir(date_path)[0])
        os.rename(file_path, date_path + '\\{}_{}_543.tif'.format(city_path.split("\\")[-1], date))

def vis_rgb_fenbu(path):
    img = io.imread(path)
    r = img[:, :, 0].reshape(-1)
    g = img[:, :, 1].reshape(-1)
    b = img[:, :, 2].reshape(-1)

    n_bins = 20
    x = [r, g, b]
    colors = ['r', 'g', 'b']

    # step2：手动创建一个figure对象，相当于一个空白的画布
    figure = plt.figure()
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # step3：在画布上添加1个子块，标定绘图位置
    axes1 = plt.subplot(1, 1, 1)
    # step4：绘制直方图
    axes1.hist(x, n_bins, histtype='bar', color=colors, label=colors)
    axes1.legend()

    # step5：展示
    plt.show()



def find_xyz(xyz,c):
    """ xyz ：该像素点的r，g，b值
        c ：三行等距列表组成的二维数组,范围 0~65536
        """
    x,y,z = 0,0,0
    if xyz[0] > c[0][0]:
        for p in range(len(c[0])):
            if xyz[0] >= c[0][p] and xyz[0] < c[0][p+1]:
                x = p + 1
                break
    if xyz[1] > c[1][0]:
        for q in range(len(c[1])):
            if xyz[1] >= c[1][q] and xyz[1] < c[1][q+1]:
                y = q + 1
                break
    if xyz[2] > c[2][0]:
        for k in range(len(c[2])):
            if xyz[2] >= c[2][k] and xyz[2] < c[2][k+1]:
                z = k + 1
                break
    return x,y,z
def rgb_fenbu(arr_rgb):
    c = np.zeros((3,split_num))
    per_box = (int)(split_range[1] / split_num)
    c0 = np.linspace(per_box,split_range[1],split_num)

    for i in range(len(c)):
        c[i] = c0

    fenbu = np.zeros((split_num,split_num,split_num))
    for i in arr_rgb:
        for j in i:
            p,q,k = find_xyz(j,c)
            fenbu[p][q][k] += 1
            #print("[{},{},{}]:{}".format(p,q,k,fenbu[p][q][k]))
    fenbu = fenbu.flatten()
    """print(fenbu.sum())
    print(fenbu.size)"""
    return fenbu

def get_fenbu(path):
    rgb_arr = io.imread(path)
    """r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    rows = len(r)
    columns = len(r[0])
    rgb_arr = np.zeros((rows,columns,3))
    for row in range(rows):
        for column in range(columns):
            rgb_arr[row][column] = [r[row][column],g[row][column],b[row][column]]"""
    fenbu = rgb_fenbu(rgb_arr)
    return fenbu


def get_KL_from_fenbu(y, x):
    """比较两个频数分布的KL散度
    """

    x = torch.tensor([x])
    y = torch.tensor([y])

    logp_x = F.log_softmax(x,dim=-1)
    p_y = F.softmax(y,dim=-1)
    KL = F.kl_div(logp_x, p_y, reduction='sum')

    """x = torch.tensor(x)
    y = torch.tensor(y)

    logp_x = F.softmax(x,dim = -1)
    p_y = F.softmax(y, dim=-1)
    KL = 0
    logp_x = logp_x.numpy()
    p_y = p_y.numpy()
    for i in range(len(logp_x)):
        if p_y[i] == 0 or logp_x[i] == 0:
            KL += logp_x
        else:
            KL += logp_x * math.log2(logp_x / p_y)"""
    return KL.item()


def path2KL(path_base, path_1):
    """ 输入两个需要比较的543图片路径
        返回KL散度
        """
    basetime = datetime.datetime.now()
    fenbu_base = get_fenbu(path_base)
    fenbu_1 = get_fenbu(path_1)
    #删除多余0值
    """delete = []
    for i in range(len(fenbu_1)):
        if fenbu_1[i] == 0 and fenbu_base[i] == 0:
            delete.append(i)
        elif fenbu_1[i] != 0 or fenbu_base[i] != 0:
            break

    fenbu_1 = np.delete(fenbu_1,delete)
    fenbu_base = np.delete(fenbu_base,delete)"""
    KL_baseto1 = get_KL_from_fenbu(fenbu_base, fenbu_1)
    now = datetime.datetime.now()
    print("one kl caculate is done.")
    print("this caculate cost {}".format(now - basetime))
    return KL_baseto1


def jijie_split(city_path, lichun, lixia, liqiu, lidong):
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
    arr = [spring, summer, autumn, winter]
    return arr


""" 二维KL散度热力图，分组多对多的比较,春-秋，夏-冬
    输入该城市某一年的文件路径
    格式为city_path/date/543.tiff
    """
def get_juzhen_from_jijie(jijie1, jijie2):
    """ 以两季列表作为输入，获得kl散度矩阵"""
    arr = np.zeros((len(jijie2), len(jijie1)), dtype=float)  # 春作列，其他季节作为行
    for i in range(len(jijie1)):
        for j in range(len(jijie2)):
            kl = path2KL(jijie1[i], jijie2[j])
            #kl_g = path2KL(jijie1[i], jijie2[j], 'g')
            #kl_b = path2KL(jijie1[i], jijie2[j], 'b')
            arr[j][i] = kl
    return arr


def relitu(siji, target, jijie1num, jijie2num, index, xp ,yp):
    r_kl_juzhen_chunqiu = get_juzhen_from_jijie(siji[jijie1num], siji[jijie2num])
    data_1 = {}
    for i in range(len(index[jijie1num])):
        data_1[index[jijie1num][i]] = r_kl_juzhen_chunqiu[:, i]
    pd_data = pd.DataFrame(data_1, index=index[jijie2num], columns=index[jijie1num])
    ax = sns.heatmap(pd_data, annot=True, fmt='.3f',vmin = 100 , vmax = 2000)
    plt.xlabel(xp, fontsize=14, color='k')  # x轴label的文本和字体大小
    plt.ylabel(yp, fontsize=14, color='k')  # y轴label的文本和字体大小
    scatter_fig = ax.get_figure()
    scatter_fig.savefig(target + "\\" + xp + "-" + yp + ".png", dpi=400)
    plt.close()


def KL_multi(city_path):
    siji = jijie_split(city_path, 204, 506, 808, 1108)

    """获取绘图坐标"""
    spring, summer, autumn, winter = [], [], [], []
    index = [spring, summer, autumn, winter]
    for i in range(len(siji)):
        for j in siji[i]:
            index[i].append((j.split("\\")[-2])[2:])

    """ 获得季节多对多kl散度矩阵
        并绘制热力图
        春-秋，夏-冬"""
    target = r'C:\Users\owl\Desktop'

    relitu(siji, target, 0, 2, index, 'spring','autumn')
    relitu(siji, target, 0, 1, index, 'spring','summer')
    relitu(siji, target, 1, 2, index, 'summer','autumn')
    relitu(siji, target, 1, 3, index, 'summer','winter')


""" 四季kl均值比较
    输入该城市某一年的文件路径
    格式为city_path/date/543.tiff
    """
def KL_avg(city_path):
    siji = jijie_split(city_path, 204, 506, 808, 1108)
    """获取四季的分布的平均值"""
    fenbu_r_avg = []
    for jijie in siji:
        arr_r = [0] * split_num * split_num * split_num
        for day in jijie:
            print("正在计算分布{}".format(day))
            r = get_fenbu(day)
            arr_r += r
        arr_r = np.asarray(arr_r)
        arr_r = arr_r / len(jijie)
        arr_r.tolist()
        print(arr_r)
        fenbu_r_avg.append(arr_r)
    kl_chunqiu_r = get_KL_from_fenbu(fenbu_r_avg[0], fenbu_r_avg[2])
    print("春-秋平均值KL散度：{}".format(kl_chunqiu_r))
    kl_chunxia_r = get_KL_from_fenbu(fenbu_r_avg[0], fenbu_r_avg[1])
    print("春-夏平均值KL散度：{}".format(kl_chunxia_r))
    kl_xiadong_r = get_KL_from_fenbu(fenbu_r_avg[1], fenbu_r_avg[3])
    print("夏-冬平均值KL散度：{}".format(kl_xiadong_r))
    kl_xiaqiu_r = get_KL_from_fenbu(fenbu_r_avg[1], fenbu_r_avg[2])
    print("夏-秋平均值KL散度：{}".format(kl_xiaqiu_r))


if __name__ == "__main__":
    root = r'C:\Users\owl\Desktop\tifftest'
    city_path = r'C:\Users\owl\Desktop\tifftest\beijing_2019'
    KL_multi(city_path)
    KL_avg(city_path)

    end = datetime.datetime.now()
    print("total time is {}".format(end - start))








