import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    src_img = cv2.imread('home.jpg',0)
    plt.imshow(src_img,cmap='gray')
    plt.savefig('grayImg.png')

    fftImg = np.fft.fft2(src_img)
    centerImg = np.fft.fftshift(fftImg)
    fftImgLog =  100 * np.log(abs(fftImg))
    centerImgLog = 100 * np.log(abs(centerImg))

    r,c = centerImgLog.shape
    ox,oy = (int)(r/2),(int)(c/2)
    
    filter1 = np.zeros((r,c),np.uint8)
    filter1= cv2.rectangle(filter1,(oy-50,ox-50),(oy+50,ox+50),(255,255,255),-1)
    imageBack1 = centerImg * filter1
    filterImg1 = np.abs(np.fft.ifft2(imageBack1))

    filter2 = np.zeros((r,c),np.uint8)
    filter2 = cv2.circle(filter2,(oy,ox),70,(255,255,255),-1)
    imageBack2 = centerImg * filter2
    filterImg2 = np.abs(np.fft.ifft2(imageBack2))

    filter1 = np.zeros((r,c),np.uint8)
    filter1= cv2.rectangle(filter1,(oy-50,ox-50),(oy+50,ox+50),(255,255,255),-1)
    imageBack1 = centerImg * filter1
    filterImg1 = np.abs(np.fft.ifft2(imageBack1))

    filter3 = np.zeros((r,c),np.uint8)
    filter3[:oy,:] = 1
    imageBack3 = centerImg * filter3
    filterImg3 = np.abs(np.fft.ifft2(imageBack3))

    filter4 = np.zeros((r,c),np.uint8)
    filter4[:ox+30,:oy+90] = 1
    imageBack4 = centerImg * filter4
    filterImg4 = np.abs(np.fft.ifft2(imageBack4))



    img_set = [src_img,fftImgLog,centerImgLog,filter1,filterImg1,filter2,filterImg2,
                filter3,filterImg3,filter4,filterImg4]
    title_set = ['Source Image','FFT Image Log','Center Image Log',
                    'Filter1','Filter1 Image','Filter2','FilterImage2',
                        'Filter3','FilterImg3','Filter4','FilterImg4']

   

    def plot_img(img_set , title_set):
        n = len(img_set)
        r,c = (3,4)
        plt.figure(figsize=(20,20))
        for i in range(n):
            plt.subplot(r,c,i+1)
            plt.imshow(img_set[i],cmap='gray')
            plt.title(title_set[i])

        plt.savefig('OutputFFT.png')
        plt.show()

    plot_img(img_set,title_set)
    

if __name__=="__main__":
   main()