import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.image as mpimg
from scipy import signal
from scipy import ndimage


def extractValueChannel(image):
    try:
        # Check if it has three channels or not 
        np.size(image, 2)
    except:
        return image
    hsvImage = col.rgb_to_hsv(image)
    return hsvImage[..., 2]



if __name__ == '__main__' :
    image = mpimg.imread("images/MainRoad.bmp")
    valueChannel = extractValueChannel(image)
    plt.figure("Original Image", figsize=(10,10))
    plt.imshow(valueChannel)
    plt.set_cmap("gray")
    #Add noise to the image
    noisy = valueChannel + 0.4*valueChannel.std() * np.random.random(valueChannel.shape)
    plt.figure("Noisy Image", figsize=(10,10))
    plt.imshow(noisy)
    plt.set_cmap("gray")
    
    
    #Now denoise the image using different smoothing filters
    w = 9
    boxFilter = np.ones((w,w)) /(w*w)
    ImBox = signal.convolve2d(noisy, boxFilter,'same')
    plt.figure("Smoothing Box",figsize=(10,10))
    plt.imshow(ImBox)
  
    sigma = 2
    gauss_denoised = ndimage.gaussian_filter(noisy, sigma)
    plt.figure("Smoothing Gauss",figsize=(10,10))
    plt.imshow(gauss_denoised)
    
    s = 5
    med_denoised = ndimage.median_filter(noisy,(s,s))
    plt.figure("Median Filter", figsize=(10,10))
    plt.imshow(med_denoised)

    
    # Edge detection 
    Sobelx = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    
    Sobely = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    lines = mpimg.imread("images/Lines.jpg")
    linesVChannel = extractValueChannel(lines)
    Ix = signal.convolve2d(linesVChannel, Sobelx)
    Iy = signal.convolve2d(linesVChannel, Sobely)
    G = np.sqrt(Ix**2 + Iy**2)
    plt.figure("Sobel Filter", figsize=(10,10))
    plt.imshow(G)
    plt.show()
    
    
    