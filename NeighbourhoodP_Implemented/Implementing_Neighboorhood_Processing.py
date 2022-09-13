import cv2
from cv2 import compare
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float


def main():
    src_img = cv2.imread('messi5.jpg',0)

    rows,cols = src_img.shape
    '''Sharpening Kernel 3 x 3'''
    kernel1 = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ],dtype=np.float32)

    '''Gaussian Blurr Kernel 5 x 5'''
    kernel2 = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ],dtype=np.float32) / 256
    

    '''This function will calculate convolution using border reflect mechanism'''
    def calc_convolution(kernel,x,y,lr,lc):
        sum = 0
        for i in range(-lr,lr+1):
            for j in range(-lc,lc+1):
                boundary_x = abs(x + i)
                boundary_y = abs(y + j)
                
                if boundary_x >= rows:
                    boundary_x = boundary_x - ((boundary_x % (rows-1))*2)
                if boundary_y >= cols:
                    boundary_y = boundary_y - ((boundary_y % (cols-1))*2)
                
                sum = sum + (kernel[i+lr,j+lc] * src_img[boundary_x,boundary_y])
        sum = np.rint(sum)
        sum = max(0,sum)
        sum = min(255,sum)
        return sum

    '''Generating each pixel for new filtering image'''
    def make_filter(Img,kernel):
        rows,cols = Img.shape
        h,w = kernel.shape
        lr,lc = int(h/2),int(w/2)
        for i in range(rows):
            for j in range(cols):
                Img[i,j] = calc_convolution(kernel,i,j,lr,lc)

    '''This section of code are used to make filter of an image with aboves two kernels'''
    implementedImg = np.zeros((rows,cols),np.uint8)
    resulting_img = cv2.filter2D(src_img,-1,kernel1)
    make_filter(implementedImg,kernel1) 

    implementedImg2 = np.zeros((rows,cols),np.float32)
    resulting_img2 = cv2.filter2D(src_img,-1,kernel2)
    make_filter(implementedImg2,kernel2)

    img_set = [src_img,implementedImg,resulting_img,implementedImg2,resulting_img2]
    title_set = ['GrayScale Image','Implemented with 3x3 kernel','Built-in with 3x3 kernel',
                'Implemented with 5x5 kernel','Built-in with 5x5 kernel']

    n = len(img_set)
    plt.figure(figsize=(20,20))

    for i in range(n):
        plt.subplot(2,3,i+1)
        plt.imshow(img_set[i],cmap='gray')
        plt.title(title_set[i])

    plt.show()

if __name__ == '__main__':
    main()
