"""
移动文件脚本 2022.01.15
1.待移动文件在同一个文件夹，并且可以根据文件名进行区分
2.目标目录包含多个文件夹
3.最终文件根据文件名存放在不同文件夹中
"""


import os
import shutil

# 想要移动文件所在的根目录
rootdir = r"F:\bda\chengdu\Landsat C2 L1 Band Files"
# 目标路径
des_path = r"F:\文件\landsat8" +"\\" + rootdir.split("\\")[-2]
# 特征名
#keyname = '190026'



# 获取目录下文件名清单
list_from = os.listdir(rootdir)
dels = []

#不选择文件名不正确的文件
"""for i in list_from:
    if keyname not in i:
        dels.append(i)
for j in dels:
    list_from.remove(j)
print(list_from)"""

#保证每个文件都有目标文件夹
for f in list_from:
    if not os.path.isdir(des_path + "//" + f.split('_')[3]):  # 判断是否有该文件夹
        os.makedirs(des_path + "//" + f.split('_')[3])
        print("已创建{}文件夹".format(des_path + "//" + f.split('_')[3]))

#获取目的地目录文件夹
list_to = os.listdir(des_path)
#print(list_to)

# 移动图片到指定文件夹
for i in range(0, len(list_from)):  # 遍历要移动目录下的所有文件
    from_path = os.path.join(rootdir, list_from[i])
    for t in range(0, len(list_to)):  #遍历目标路径文件夹
        final_path = os.path.join(des_path, list_to[t])  # 最终目标目录
        if list_from[i].split('_')[3] == list_to[t]:    #判断文件名与目录名是否匹配
            #if os.path.exists(final_path + "//" + list_from[i]):     #如果该文件已存在在目标目录，则删除该文件
                #os.remove(from_path)
                #print("{}已存在".format(list_from[i]))
            #else:
                try:
                    shutil.move(from_path, final_path)  # 移动文件到目标路径
                except:
                    pass
                print("{}已移动到{}".format(list_from[i],final_path))
