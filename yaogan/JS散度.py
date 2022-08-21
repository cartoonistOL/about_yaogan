import math
import os
import scipy.stats
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import datetime

import pandas as pd


from skimage import io
import seaborn as sns

#分割范围
from 波段合成 import GRID

split_range = (0, 256 * 256)
#每个通道分割数量
split_num = 128



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
    """ 寻找像素点所处的三维数组坐标
        xyz ：该像素点的r，g，b值
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

def find_5432(xyz,c):
    """ 寻找像素点所处的三维数组坐标
        xyz ：该像素点的r，g，b值
        c ：三行等距列表组成的二维数组,范围 0~65536
        """
    x,y,z,v = 0,0,0,0
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
    if xyz[3] > c[3][0]:
        for k in range(len(c[3])):
            if xyz[3] >= c[3][k] and xyz[3] < c[3][k+1]:
                v = k + 1
                break
    return x,y,z,v

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
    fenbu = fenbu.flatten().squeeze()
    #print(fenbu.sum())
    #print(fenbu.size)
    return fenbu

def get_fenbu(path):
    rgb_arr = io.imread(path)[:224,:224,:]
    fenbu = rgb_fenbu(rgb_arr)
    return fenbu

def get_fenbu_5432(path):
    arr_5432 = io.imread(path)[:224, :224, :]
    c = np.zeros((4, split_num))
    per_box = (int)(split_range[1] / split_num)
    c0 = np.linspace(per_box, split_range[1], split_num)

    for i in range(len(c)):
        c[i] = c0

    fenbu = np.zeros((split_num, split_num, split_num,split_num))
    for i in arr_5432:
        for j in i:
            p, q, k, v = find_5432(j, c)
            fenbu[p][q][k][v] += 1
            # print("[{},{},{}]:{}".format(p,q,k,fenbu[p][q][k]))

    fenbu = fenbu.flatten().squeeze()
    # print(fenbu.sum())
    # print(fenbu.size)
    return fenbu


def get_KL_from_fenbu(y, x):
    """比较两个频数分布的KL散度
    """

    x = torch.squeeze(torch.tensor(x))
    y = torch.squeeze(torch.tensor(y))

    #logp_x = F.log_softmax(x,dim = 0)
    #p_y = F.softmax(y,dim = 0) #一维时使用dim=0

    KL = scipy.stats.entropy(y,x)
    #KL = F.kl_div(logp_x, p_y, reduction='sum')

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
    return KL


def path2JS(path_base, path_1):
    """ 输入两个需要比较的543图片路径
        返回JS散度
        """
    basetime = datetime.datetime.now()
    fenbu_base = get_fenbu(path_base)
    fenbu_1 = get_fenbu(path_1)
    JS_baseto1 = get_JS_from_fenbu(fenbu_base, fenbu_1)
    now = datetime.datetime.now()
    print("one js caculate is done.")
    print(JS_baseto1)
    print("this caculate cost {}".format(now - basetime))
    return JS_baseto1


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
def get_js_from_jijie(jijie1, jijie2):
    """ 以两季列表作为输入，获得js散度矩阵"""
    arr = np.zeros((len(jijie2), len(jijie1)), dtype=float)  # 春作列，其他季节作为行
    for i in range(len(jijie1)):
        for j in range(len(jijie2)):
            js = path2JS(jijie1[i], jijie2[j])

            arr[j][i] = js
    return arr

def path2KL(path_base, path_1):
    """ 输入两个需要比较的543图片路径
        返回KL散度
        """
    basetime = datetime.datetime.now()
    fenbu_base = get_fenbu(path_base)
    fenbu_1 = get_fenbu(path_1)
    KL_baseto1 = get_KL_from_fenbu(fenbu_base, fenbu_1)
    now = datetime.datetime.now()
    print("one kl caculate is done.")
    print("this caculate cost {}".format(now - basetime))
    return KL_baseto1

def get_kl_from_jijie(jijie1, jijie2):
    """ 以两季列表作为输入，获得kl散度矩阵"""
    print("===============计算矩阵=============")
    arr = np.zeros((len(jijie2), len(jijie1)), dtype=float)  # 春作列，其他季节作为行
    for i in range(len(jijie1)):
        for j in range(len(jijie2)):
            js = path2KL(jijie1[i], jijie2[j])

            arr[j][i] = js
    return arr


def relitu(siji, target, jijie1num, jijie2num, index, xp ,yp):
    print("正在绘制热力图")
    js_juzhen = get_js_from_jijie(siji[jijie1num], siji[jijie2num])
    data_1 = {}
    for i in range(len(index[jijie1num])):
        data_1[index[jijie1num][i]] = js_juzhen[:, i]
    pd_data = pd.DataFrame(data_1, index=index[jijie2num], columns=index[jijie1num])
    ax = sns.heatmap(pd_data, annot=True, fmt='.3f',vmin = 0. , vmax = 1.)
    plt.xlabel(xp, fontsize=14, color='k')  # x轴label的文本和字体大小
    plt.ylabel(yp, fontsize=14, color='k')  # y轴label的文本和字体大小
    scatter_fig = ax.get_figure()
    scatter_fig.savefig(target + "\\" + xp + "-" + yp + ".png", dpi=400)
    plt.close()
    print("==========one pic is done=============")


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

def get_JS_from_fenbu(p,q):
    av = 0.5*(p+q)
    return 0.5*(get_KL_from_fenbu(p,av))+0.5*(get_KL_from_fenbu(q,av))

def combine(root,des_date_path):
    run = GRID()
    bands = os.listdir(root)
    band1 = bands[0]
    band2 = bands[1]
    band3 = bands[2]
    proj, geotrans, data2, row1, column1 = run.read_img(root + '\\' + band1)  # 读数据
    proj, geotrans, data3, row2, column2 = run.read_img(root + '\\' + band2)  # 读数据
    proj, geotrans, data4, row3, column3 = run.read_img(root + '\\' + band3)  # 读数据
    data_combine = np.array((data4, data3, data2,), dtype=data2.dtype)  # 按序将3个波段像元值放入
    run.write_img(des_date_path + '\\' + root.split("\\")[-1] + '_combine_432.tif', proj, geotrans, data_combine)  # 写为3波段数据
    print('{}已保存'.format(des_date_path + '\\combine_432.tif'))

def js_4days(paths):
    arr = np.zeros((len(paths),len(paths)))
    name = ['']*4
    name_1 = [''] * 4
    for n in range(len(paths)):
        name[n] = paths[n].split('_')[-2][-4:]
        name_1[len(paths) - 1 - n] = name[n]
    print(name,name_1)
    for i in range(len(paths)):
        for j in range(len(paths)):
            arr[i][len(paths) - 1 - j] = path2JS(paths[i],paths[j])
    data = pd.DataFrame(arr,index=name,columns=name_1)
    ax = sns.heatmap(data, annot=True, fmt='.3f', vmin=0., vmax=1.)
    scatter_fig = ax.get_figure()
    scatter_fig.savefig(r'C:\Users\owl\Desktop\4days_JS_pic.png', dpi=400)
    plt.close()

if __name__ == "__main__":
    start = datetime.datetime.now()
    root = r'C:\Users\owl\Desktop\tifftest'
    city_path = r'C:\Users\owl\Desktop\tifftest\bejing2019_5432'
    #KL_multi(city_path)
    #KL_avg(city_path)
    paths_4days = [r'C:\Users\owl\Desktop\tifftest\beijing_2019\20190326\beijing_2019_20190326_543.tif',
                   r'C:\Users\owl\Desktop\tifftest\beijing_2019\20190614\beijing_2019_20190614_543.tif',
                   r'C:\Users\owl\Desktop\tifftest\beijing_2019\20190918\beijing_2019_20190918_543.tif',
                   r'C:\Users\owl\Desktop\tifftest\beijing_2019\20200108\beijing_2019_20200108_543.tif',]
    js_4days(paths_4days)
    #a = r'C:\Users\owl\Desktop\tifftest\20130512_combine_432.tif'
    #b = r'C:\Users\owl\Desktop\tifftest\20130613_combine_432.tif'
    #c = r'C:\Users\owl\Desktop\tifftest\20131206_combine_432.tif'

    #a1 = path2JS(a,b)
    #b1 = path2JS(a,c)
    #a2 = path2JS(b,a)
    #print(a1,a2)

    end = datetime.datetime.now()
    print("total time is {}".format(end - start))

    #combine(r'F:\文件\croped_city\beijing\20130613',r'C:\Users\owl\Desktop\tifftest')








