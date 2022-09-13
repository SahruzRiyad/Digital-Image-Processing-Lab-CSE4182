import cv2
import matplotlib.pyplot as plt
import numpy as np

th = cv2.imread('./LinuxLogo.jpg',0)
# _,th = cv2.threshold(src_img,127,255,cv2.THRESH_BINARY)

kernel1 = np.ones((5, 5), np.uint8)
kernel2 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]], dtype=np.uint8)
kernel3 = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]], dtype=np.uint8)

img_dilate1 = cv2.dilate(th,kernel1,iterations=1)
img_erosion1 = cv2.erode(th,kernel1,iterations=1)
img_open1 = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel1)
img_close1 = cv2.morphologyEx(th,cv2.MORPH_CLOSE,kernel1)

img_dilate2 = cv2.dilate(th,kernel2,iterations=1)
img_erosion2 = cv2.erode(th,kernel2,iterations=1)
img_open2 = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel2)
img_close2 = cv2.morphologyEx(th,cv2.MORPH_CLOSE,kernel2)

img_dilate3 = cv2.dilate(th,kernel3,iterations=1)
img_erosion3 = cv2.erode(th,kernel3,iterations=1)
img_open3 = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel3)
img_close3 = cv2.morphologyEx(th,cv2.MORPH_CLOSE,kernel3)

img_set = [img_dilate1,img_erosion1,img_open1,img_close1,
            img_dilate2,img_erosion2,img_open2,img_close2,
                img_dilate3,img_erosion3,img_open3,img_close3]

title_set = ['Dilation1','Erosion1','Open1','Close1',
                'Dilation2','Erosion2','Open2','Close2',
                    'Dilation3','Erosion3','Open3','Close3']

def plot_img(img_set,title_set):
    n = len(img_set)
    r,c = 3,4
    plt.figure(figsize=(20,20))
    for i in range(n):
        plt.subplot(r,c,i+1)
        plt.imshow(img_set[i],cmap='gray')
        plt.title(title_set[i])
    plt.savefig('Output.png')
    plt.show()

plot_img(img_set,title_set)

