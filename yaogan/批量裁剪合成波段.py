import os

from 裁剪 import crop
from 波段合成 import combine



if __name__ == '__main__':
    root_path = r'C:\Users\owl\Desktop\testtif'     #根目录，该目录下包含各个城市文件夹

    crop_path = r'C:\Users\owl\Desktop\croped_city'
    combine_path = r'C:\Users\owl\Desktop\combine_city'
    dicOfcity = {'beijing': [116.0630,40.1869,116.7300,39.6824],'berlin': [13.0558,52.68,13.7780,52.3375],
                 'budapest':[18.9066,47.6088,19.3621,47.3349],'brussel':[4.3069,50.9159,4.4383,50.7935]}
    crop(root_path,crop_path,dicOfcity)
    combine(crop_path,combine_path, dicOfcity)







