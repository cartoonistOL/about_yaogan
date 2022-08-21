
from osgeo import gdal

img_path = r'C:\Users\owl\Desktop\beijing_20130512_5432.tif'

def to224(img_path):
    dataset = gdal.Open(img_path)
    x_pixel = dataset.RasterXSize
    y_pixel = dataset.RasterYSize
    #print(x_pixel, y_pixel)
    j = 0
    flag = 0
    lie_2 = 0
    while (lie_2 != y_pixel):
        i = 0
        hang_1, lie_1 = 0, 0
        hang_2, lie_2 = 0, 0
        while (hang_2 != x_pixel):
            if flag == 0:
                hang_1, lie_1 = [i * 224, j * 224]
                hang_2, lie_2 = [(i + 1) * 224, (j + 1) * 224]
            elif flag == 1:
                hang_1, lie_1 = [i * 224, y_pixel - 224]
                hang_2, lie_2 = [(i + 1) * 224, y_pixel]
            if x_pixel - hang_1 < 224:
                hang_1 = x_pixel - 224
                hang_2 = x_pixel
            target = r'C:\Users\owl\Desktop\out' + '\\' + '{}_{}.tif'.format(i, j)
            src: gdal.Dataset = dataset
            gdal.Translate(target, src, srcWin=[lie_1, hang_1, lie_2 - lie_1, hang_2 - hang_1],
                           options=['-a_scale', '1'])
            print("{}".format(target))
            i += 1
        j += 1
        if (y_pixel - (j * 224) < 224):
            flag = 1