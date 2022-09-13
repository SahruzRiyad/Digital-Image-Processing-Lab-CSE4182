import matplotlib.pyplot as plt
import cv2
import numpy as np

src_img = cv2.imread('rgbImg2.jpg',0)
img_hist = cv2.calcHist([src_img],[0],None,[256],[0,256])
rows,cols= src_img.shape
print(img_hist.shape)
hist = np.zeros((256,1),np.uint)

for x in range(rows):
    for y in range(cols):
        pixel = src_img[x,y]
        hist[pixel] = hist[pixel] + 1

for i in range(256):
    if hist[i] != img_hist[i]:
        print('Has difference')


plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.hist(hist,256,[0,256])
plt.title('Implemented Histogram')

plt.subplot(1,2,2)
plt.hist(img_hist,256,[0,256])
plt.title('Built-in Histogram')

plt.show()

