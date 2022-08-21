import io

import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
import geopandas

import PIL.Image as Image
from matplotlib import pyplot as plt

def show(p1):
    p1.plot(column='building')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # plt.gca()表示获取当前子图"Get Current Axes"。
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.axis('off') # 关掉坐标轴为 off
    plt.margins(0, 0)

    buffer = io.BytesIO()
    plt.savefig(buffer,format = 'png')
    #用PIL或CV2从内存中读取
    dataPIL = Image.open(buffer)
    #转换为nparrary
    data = np.asarray(dataPIL)
    print(data.shape)
    #释放缓存
    buffer.close()
    img = Image.fromarray(data)
    img.show()
# ox.geometries.geometries_from_xml返回geopandas.GeoDataFrame对象
p1 = ox.geometries.geometries_from_xml("map.osm")
buildings = p1["building"]
show(p1)
# p1.plot(column='building')

# plt.show()
# print(p1)



# ds = p1.loc['way']['geometry']
# fig = plt.figure(dpi=400)
# ax = plt.subplot()
# ds.plot(edgecolor = 'k',alpha = 0.5, ax = ax)
# plt.show()
