"""从filter.json中得到云量小于5的影像列表，然后根据日期筛选现有影像"""

import json
import os
import shutil

# top = r"F:\file\croped_city"  # 待筛选根目录
target_dir = r"F:\file\croped_city_5cloud"    # 目标目录

# cities = os.listdir(top)
filterfile = r'eu_filter.json'
filter = []
count = 0
total = 0
with open(filterfile) as f:
    filter = json.load(f)
    print(filter.keys())
    print(len(filter.keys()))
a = 0
# for key in filter.keys():
#     citypath = os.path.join(top,key)
#     dates = os.listdir(citypath)
#     total += len(dates)
#     for date in dates:
#         filter_dates = [i.split("_")[3] for i in filter[key]]
#         if date in filter_dates:
#             count += 1
#             target = os.path.join(target_dir + "\\" + key,date)
#             if not os.path.exists(target):
#                 shutil.copytree(os.path.join(citypath,date),target)  # 目标目录不能存在
#                 print("copy to " + os.path.join(target + "\\" + key,date))
#             else:
#                 print(target + "已存在" )
# print("count:"+str(count))
# print("total:"+str(total))