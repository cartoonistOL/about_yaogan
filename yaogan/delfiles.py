import os


def delfiles(top):
    num = 0
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            if '432' in name:
                file_path = os.path.join(root, name)
                print("{}已删除".format(file_path))
                os.remove(file_path)
                num += 1
    print("共删除{}个文件".format(num))

if __name__ == '__main__':
    top = r'C:\Users\owl\Desktop\tifftest\beijing_2019'
    delfiles(top)