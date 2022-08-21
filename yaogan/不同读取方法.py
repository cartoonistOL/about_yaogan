
import cv2
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal_array
from skimage import io
bath = r'C:\Users\owl\Desktop\123\blue20201225_1061555.tif'
img_cv = cv2.imread(bath)#能读tif，但是如果是3个通道，就会被强制变成三个一样的通道
img_cv2 = io.imread(bath)#[h,w,c]格式
img_cv3 = gdal_array.LoadFile(bath)#[c,h,w]格式
print(img_cv)
print(img_cv.shape)
print(img_cv2)
print(img_cv2.shape)
print(img_cv3)
print(img_cv3.shape)

'''B3_np = np.array(B3_gdal)
print(B3_np)'''
print('-----------------------------------------------------------------')
'''plt.imshow(img_cv)
plt.show()
print(img_cv)
print("img_cv:",img_cv.shape)'''
