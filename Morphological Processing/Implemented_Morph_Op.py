import cv2
import matplotlib.pyplot as plt
import numpy as np

src_img = np.array([
    [1,1,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,1,0,1,0,0,0,1],
    [0,0,1,0,1,0,1,0],
    [0,0,0,1,1,1,0,0],
    [0,1,0,0,1,0,1,0],
    [0,0,1,1,1,1,0,0]
],dtype=np.uint8)

structuring_kernel = np.array([
    [0,1,0],
    [1,1,1],
    [0,1,0]
],np.uint8)

def getNumberOfOnes(kernel):
    r,c = kernel.shape
    count = 0
    for i in range(r):
        for j in range(c):
            if(kernel[i,j] == 1):
                count = count + 1
    return count

def calc_morphOp_value(src_img,kernel,ox,oy):
        sum = 0
        rows,cols = kernel.shape
        r,c = src_img.shape
        lr,lc = int(rows/2) , int(cols/2)
        for i in range(-lr,lr+1):
            for j in range(-lc,lc+1):
                x = abs(ox + i)
                y = abs(oy + j)
                if x >= r:
                    x = x - (x % (r-1))*2
                if y >= c:
                    y = y - (y % (c-1))*2
                if kernel[i+lr,j+lc] == 1:
                    sum = sum + (kernel[i+lr,j+lc] & src_img[x,y])
        return sum

def make_dilate(src_img,kernel):
    r,c = src_img.shape
    img = np.zeros((r,c),np.uint8)
    for i in range(r):
        for j in range(c):
            x = calc_morphOp_value(src_img,kernel,i,j)
            if x >= 1:
                img[i,j] = 1
            else:
                img[i,j] = 0

    return img

def make_erode(src_img,kernel):
    r,c = src_img.shape
    img = np.zeros((r,c),np.uint8)
    ones = getNumberOfOnes(kernel)
    for i in range(r):
        for j in range(c):
            x = calc_morphOp_value(src_img,kernel,i,j)
            if x == ones:
                img[i,j] = 1
            else:
                img[i,j] = 0

    return img

def make_open(src_img,kernel):
    img = make_erode(src_img,kernel)
    img = make_dilate(img,kernel)
    return img

def make_close(src_img,kernel):
    img = make_dilate(src_img,kernel)
    img = make_erode(img,kernel)
    return img

dilate1  = make_dilate(src_img,structuring_kernel)
print(dilate1)
erode1 = make_erode(src_img,structuring_kernel)
open1 = make_open(src_img,structuring_kernel)
close1 = make_close(src_img,structuring_kernel)

OpenCVdilate1 = cv2.dilate(src_img,structuring_kernel,iterations=1)
OpenCVerode1 = cv2.erode(src_img,structuring_kernel,iterations=1)
OpenCVopen = cv2.morphologyEx(src_img,cv2.MORPH_OPEN,structuring_kernel)
OpenCVclose = cv2.morphologyEx(src_img,cv2.MORPH_CLOSE,structuring_kernel)

img_set = [src_img,OpenCVdilate1,dilate1,OpenCVerode1,erode1,OpenCVopen,open1,
            OpenCVclose,close1]
title_set = ['Source Image','OpenCV Dilation','Implemented Dilation','OpenCV Erosion',
                'Implemented Erostion','OpenCV Open','Implemented Open','OpenCV Close',
                    'Implemented Close']

def plot_img(img_set,title_set):
    n = len(img_set)
    r,c = 3,3
    plt.figure(figsize=(20,20))
    for i in range(n):
        plt.subplot(r,c,i+1)
        plt.imshow(img_set[i],cmap='gray')
        plt.title(title_set[i])

    plt.savefig('outputImg.png')
    plt.show()

plot_img(img_set,title_set)
