'''This program is for adding Noise(salt-whilte spots on the 
dark region and pepper-black spots on the white region'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

src_img = cv2.imread('salt-pepper.jpeg',0)
kernel1 = np.ones((3,3),dtype=np.float32) / 9
kernel2 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
],dtype=np.float32) / 16

# kernel3 = np.array([
#     [2, 2, 4],
#     [2, 9, 5],
#     [1, 2, 4]
# ],dtype=np.float32)

rows,cols = src_img.shape

noise_img = np.copy(src_img)

x = np.random.randint(2,size=100000) * 255

for i in range(x.size):
    xCord = np.random.randint(0,rows-1)
    yCord = np.random.randint(0,cols-1)

    noise_img[xCord,yCord] = x[i]

blurr_img = cv2.filter2D(noise_img,-1,kernel1)
gaussian_img = cv2.filter2D(noise_img,-1,kernel2)
median_img = cv2.medianBlur(noise_img,3)

img_set = [src_img,noise_img,blurr_img,gaussian_img,median_img]
title_set = ['Source Image','Noise Image','Averaging Filter Image',
            'Gaussian Filter Image','Median Filter Image']

n = len(img_set)

for i in range(n):
    plt.subplot(2,3,i+1)
    plt.imshow(img_set[i],cmap='gray')
    plt.title(title_set[i])

plt.savefig('NoiseFig.png')
plt.show()

