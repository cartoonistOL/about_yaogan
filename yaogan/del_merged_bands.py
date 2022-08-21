import os
import shutil
import time


"""删除相关文件"""
root_path = r'F:\文件\landsat8'
out_path = r'F:\文件\crop_2345band'
citys = os.listdir(root_path)

del_date = []
del_date_outpath = []
datels = []

right = {"shanghai":[],"changsha":["123040","123041"],"hefei":["121037","121038"],
         "nanjing":["120037","120038"],"shenyang":["119030","119031"],
         "xian":["127036","127037"]}   # 正确的行列号
for city in citys:
    if city != list(right.keys())[5]:
        continue
    print(city, "is working...")
    time1 = time.time()
    city_path = root_path + "\\" + city
    dates = os.listdir(city_path)
    for date in dates:
        sum = 0
        date_path = city_path + "\\" + date
        files = os.listdir(date_path)
        for file in files:
            if (right[city][0] in file) or (right[city][1] in file):
                sum += 1
        # 两个影像拼接，正常会有8个波段
        if sum < 8:
            datels.append(date)
            del_date.append(date_path)
            del_date_outpath.append(out_path + "\\" + city + "\\" + date)

del_date = list(set(del_date))
del_date_outpath = list(set(del_date_outpath))
print(del_date)
print(del_date_outpath)
datels = list(set(datels))
print(datels)
print(len(datels))
#print(1 if "20131005" in datels else 0)

"""for p in del_date:
    try:
        shutil.rmtree(p)
    except:
        pass"""
for q in del_date_outpath:
    try:
        print(q if os.path.exists(q) else "")
        #shutil.rmtree(q)
    except:
        pass