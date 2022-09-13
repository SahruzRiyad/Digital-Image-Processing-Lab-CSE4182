import cv2
import matplotlib.pyplot as plt
import numpy as np

src_img = plt.imread('LinuxLogo.jpg')
src_img = cv2.cvtColor(src_img,cv2.COLOR_RGB2GRAY)
h,w = src_img.shape
mask_img = np.zeros((h,w),np.uint8)
mask_img = cv2.rectangle(mask_img,(0,0),(230,320),(255,255,255),-1)

bitwiseAnd = cv2.bitwise_and(mask_img,src_img)
bitwiseOr = cv2.bitwise_or(mask_img,src_img)
bitwiseNot = cv2.bitwise_not(src_img)
bitwiseXor = cv2.bitwise_xor(mask_img,src_img)


img_set = [src_img,mask_img,bitwiseAnd,bitwiseOr,bitwiseNot,bitwiseXor]
title_set = ['Original Image','Mask Image','Bitwise AND','Bitwise OR','Bitwise NOT','Bitwise XOR']

for i in range(len(img_set)):
    plt.subplot(2,3,i+1)
    plt.imshow(img_set[i],cmap='gray')
    plt.title(title_set[i])
    
plt.show()