import PIL.Image as Image
import numpy as np

def get_mask_ls(mask_np,row,col,img_size=224,batch_size=16,rate=0.4):
    target_np = mask_np[(row - 1)*img_size:row*img_size,(col - 1)*img_size:col*img_size]
    h_num = w_num = int(img_size / batch_size)
    mk_ls = []
    for i in range(h_num):
        for j in range(w_num):
            clip = target_np[i*h_num:(i+1)*h_num,j*w_num:(j+1)*w_num]
             # """根据阈值rate判断矩阵是否属于植被,1代表植被,0代表非植被"""
            if (np.count_nonzero(clip == 1)) / int(clip.size) > rate:
                mk_ls.append(1)
            else:
                mk_ls.append(0)
    return mk_ls


img = Image.open(r"C:\Users\owl\Desktop\masked\masked_beijing.jpg")

mk = np.zeros((img.height,img.width))
img = np.asarray(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if (img[i,j,:] == 255).all():
            mk[i,j] = 0
        else:
            mk[i,j] = 1

t = get_mask_ls(mk,row=1,col=1)
print(t)