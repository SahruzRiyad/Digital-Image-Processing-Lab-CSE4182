import numpy as np
import matplotlib.pyplot as plt
import cv2

def MorphCalc(mat,ker):
   
    def dilate(mat,ker):
        r,c = mat.shape
        new_img = np.zeros((r,c),dtype = np.uint8)
        img = cv2.copyMakeBorder(mat, 1, 1 , 1, 1 , borderType=cv2.BORDER_DEFAULT)
        for i in range(r):
            for j in range(c):
                temp = np.sum(np.multiply(img[i:ker.shape[0]+i,j:ker.shape[1]+j],ker))

                if temp >= 1:
                    new_img[i,j] = 1 

        return new_img
    
    def erode(mat,ker):
        r,c = mat.shape
        new_img = np.zeros((r,c),dtype = np.uint8)
        img = cv2.copyMakeBorder(mat, 1, 1 , 1, 1 , borderType=cv2.BORDER_DEFAULT)
        ones = (ker == 1).sum()

        for i in range(r):
            for j in range(c):
                temp = np.sum(np.multiply(img[i:ker.shape[0]+i,j:ker.shape[1]+j],ker))

                if temp == ones:
                    new_img[i,j] = 1 

        return new_img
    
    def Open(mat,ker):
        img = erode(mat,ker)
        new_img = dilate(img,ker)

        return new_img

    def Close(mat,ker):
        img = dilate(mat,ker)
        new_img = erode(img,ker)
        
        return new_img

    dilateImg = dilate(mat,ker)
    erodeImg = erode(mat,ker)
    openImg  = Open(mat,ker)
    closeImg = Close(mat,ker)

    img_set = [mat,dilateImg,erodeImg,openImg,closeImg]
    title_set = ['Source' , 'Dilatiion' , 'Erosion' , 'Open' , 'Close']

    return img_set , title_set


def main():
    img_path = np.array([
    [1,1,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,1,0,1,0,0,0,1],
    [0,0,1,0,1,0,1,0],
    [0,0,0,1,1,1,0,0],
    [0,1,0,0,1,0,1,0],
    [0,0,1,1,1,1,0,0]
],dtype=np.uint8)

    # rgb = plt.imread(img_path)
    # gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    structuring_kernel = np.array([
    [0,1,0,1,0],
    [0,1,0,1,0],
    [0,1,0,1,0],
    [0,1,0,1,0],
    [0,1,0,1,0],
],np.uint8)

    img_set , title_set = MorphCalc(img_path,structuring_kernel)
    
    def plot_img(img_set,title_set):
        n = len(img_set)
        r,c = 2,3
        plt.figure(figsize=(20,20))
        for i in range(n):
            plt.subplot(r,c,i+1)
            plt.imshow(img_set[i],cmap='gray')
            plt.title(title_set[i])

        plt.show()
    
    plot_img(img_set,title_set)

if __name__ == '__main__':
    main()