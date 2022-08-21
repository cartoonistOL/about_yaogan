import os

def ComposeisRight(city_path):
    faul_count = 0
    faul_format = 0 
    for file in os.listdir(city_path):
        date_path = os.path.join(city_path, file)
        filelist = os.listdir(date_path)
        for filename in filelist:
            if 'refl' in filename:
                os.remove(date_path + '\\' + filename)
        if len(filelist) != 4:
            faul_count += 1
            print(date_path + '中数量不对，为{}'.format(len(filelist)))
        else:
            if not 'B2' in filelist[0] and 'B3' in filelist[1] and 'B4' in filelist[2] and 'B5' in filelist[3]:
                faul_format += 1
                print("{}中文件格式不正确".format(date_path))
    print('{}中{}个问题'.format(city_path,faul_count + faul_format))
    if faul_count + faul_format == 0:
        return 1
    else:
        return 0

def CombineisRight(city_path,channels_num):
    faul_count = 0
    faul_format = 0
    for file in os.listdir(city_path):
        filelist = os.listdir(os.path.join(city_path, file))
        date_path = os.path.join(city_path, file)
        if len(filelist) != channels_num:
            faul_count += 1
            print(date_path + '中数量不对')
        else:
            filelist = os.listdir(date_path)
            if not '432' in filelist[0] and '543' in filelist[1]:
                faul_format += 1
                print("{}中文件格式不正确".format(date_path))
    print('{}中{}个问题'.format(city_path,faul_count + faul_format))
    if faul_count + faul_format == 0:
        return 1
    else:
        return 0

if __name__ == '__main__':
    city_path = r'F:\文件\europe\vienna'
    ComposeisRight(city_path)
