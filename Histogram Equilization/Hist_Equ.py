import cv2
import matplotlib.pyplot as plt
import numpy as np

src_img = cv2.imread('./scenary2.jpeg',0)

plt.imshow(src_img,cmap='gray')
plt.savefig('inputImg.png')
plt.show()

src_img_hist = cv2.calcHist([src_img],[0],None,[256],[0,256])
hist_equ_cv2 = cv2.equalizeHist(src_img)
L = pow(2,8)

def doHistEqu(img,histogram,L):
    CDF = histogram.cumsum()
    CDFmin = CDF.min()
    r,c = img.shape
    size = r * c
    newImg = np.zeros((r,c),np.uint8)
    for x in range(r):
        for y in range(c):
            newImg[x,y] = ((CDF[img[x,y]] - CDFmin) / (size - CDFmin)) * (L-1)

    return newImg

equ_img = doHistEqu(src_img,src_img_hist,L)
img_set = [src_img,hist_equ_cv2,equ_img]
title_set = ['Source Image','Source Image Histogram','OpenCV Equalized Image',
                'OpenCV Equalized Image Histogram','Implemented Equalized Image',
                    'Implemented Equalized Image Histogram']

def plot_img(img_set,title_set):
    n = len(img_set)
    r,c = 3,2
    plt.figure(figsize=(20,20))
    for i in range(n):
        plt.subplot(r,c,i*2+1)
        plt.imshow(img_set[i],cmap='gray')
        plt.title(title_set[i*2])
        plt.subplot(r,c,i*2+2)
        plt.hist(img_set[i].flatten(),256,[0,256])
        plt.title(title_set[i*2+1])
    plt.savefig('HistEquOpenCV.png')   
    plt.show()

plot_img(img_set,title_set)

