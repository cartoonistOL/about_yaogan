import os
import shutil
import sys


def copy(come,to):
    try:
        shutil.copy(come, to)
    except IOError as e:
        print("Unable to copy file. %s" % e)
    except:
        print("Unexpected error:", sys.exc_info())

def div_siji(city_path):
    lichun,lixia,liqiu,lidong= 204, 506, 808, 1108
    dates = os.listdir(city_path)
    for date in dates:
        date_path = os.path.join(city_path,date)
        file = os.listdir(date_path)[0]
        file_path = os.path.join(date_path,file)

        day = (int)(date[4:8])
        #print(day)
        if day >= lichun and day < lixia:
            pass
        elif day >= lixia and day < liqiu:
            pass
        elif day >= liqiu and day < lidong:
            copy(file_path,r'F:\文件\combine_siji_vienna_ex\beijing\autumn' + "\\" + file_path.split('\\')[-1])
        elif day >= lidong or day < lichun:
            pass


if __name__ == '__main__':
    #city = ['beijing','berlin','bremen','budapest','dublin','london','riga','roma','vienna']
    city = ['vienna']
    root_path = r'F:\文件\combine5432'
    for i in range(len(city)):
        city_path = root_path + "\\" + city[i]
        div_siji(city_path)
