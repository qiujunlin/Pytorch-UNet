import  torch
from PIL import Image
import  numpy as np

import matplotlib.pyplot as plt
import cv2
a= Image.open("F:\dataset\competition_data/train\images/000e218f21.png")
b=Image.open("F:\dataset\carvana-image-masking-challenge/train_masks/0cdf5b5d0ce1_01_mask.gif")
#print(a.size) # 返回的是图片的高和宽度
a=a.convert('RGB') # 减去多余的通道  不然抓换后的的形状为（x,x,4）
d=a.resize(200,200)
#b=b.convert("RGB")
a=np.array(b)
print(b.mode)
print(a.shape)
#print(a[660,1000:,:])
c=cv2.imread("F:\dataset\carvana-image-masking-challenge/train_masks/0cdf5b5d0ce1_01_mask.gif")
print(c.shape)

#np.array(np.array(a))
