import matplotlib.pyplot as plt
import numpy as np
import cv2

src_image = plt.imread('messi5.jpg')
src_image = cv2.cvtColor(src_image,cv2.COLOR_RGB2GRAY)

h,w = src_image.shape
plainImg = np.zeros((8,h,w))

for i in range(h):
    for j in range(w):
        for k in range(8):
            pixel = src_image[i,j]
            x = bool(pixel & (1<<k))
            plainImg[k,i,j] = x

plt.figure(figsize=(20,20))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(plainImg[i],cmap='gray')
    plt.title('Bit-Pos '+str(i))

plt.show()
