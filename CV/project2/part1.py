from scipy.signal import convolve2d

import numpy as np
import cv2
import matplotlib.pyplot as plt



### Parameters ###
THRESHOLD = 25

### Part 1-1: Finite Difference Operator ### 
def dx_filter(img):
    dx = np.array([[1 ,-1]])
    return convolve2d(img, dx, mode='same', boundary='symm')

def dy_filter(img):
    dy = np.array([[1], [-1]])
    return convolve2d(img, dy, mode='same', boundary='symm')

def binarize(img, threshold):
    return np.where(img > threshold, 255, 0)

def part1_1():
    input_path = "./data/cameraman.png"
    in_path = "./output/part1/origin/input.png"
    gx_path = "./output/part1/origin/gx.png"
    gy_path = "./output/part1/origin/gy.png"
    gm_path = "./output/part1/origin/gm.png"
    bin_path = "./output/part1/origin/binarized.png"

    img = cv2.imread(input_path, 0)
    img_gx = dx_filter(img)
    img_gy = dy_filter(img)
    img_gm = np.sqrt(img_gx**2 + img_gy**2)
    
    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(10, 6))
    ax_x.imshow(img_gx, cmap='gray')
    ax_x.set_title('Gradient X')
    ax_x.set_axis_off()
    ax_y.imshow(img_gy, cmap='gray')
    ax_y.set_title('Gradient Y')
    ax_y.set_axis_off()
    plt.savefig('./output/part1/origin/gradient.png')

    # Binarize the images
    img_bin = binarize(img_gm, THRESHOLD)

    # Save original, gradient mag, and gradient binarized
    fig, (ax_orig, ax_mag, ax_bin) = plt.subplots(1, 3, figsize=(15, 6))
    ax_orig.imshow(img, cmap='gray')
    ax_orig.set_title('Original')
    ax_orig.set_axis_off()

    ax_mag.imshow(img_gm, cmap='gray')
    ax_mag.set_title('Gradient magnitude')
    ax_mag.set_axis_off()

    ax_bin.imshow(img_bin, cmap='gray')
    ax_bin.set_title('Gradient binarize')
    ax_bin.set_axis_off()
    plt.savefig('./output/part1/origin/gradient_binarized.png')

    cv2.imwrite(in_path, img)
    cv2.imwrite(gx_path, img_gx)
    cv2.imwrite(gy_path, img_gy)
    cv2.imwrite(gm_path, img_gm)
    cv2.imwrite(bin_path, img_bin)

### Part 1-2: Derivative of Gaussian (DoG) Filter ### 
def gaussian_filter(img, size=7, sigma=1):
    g = cv2.getGaussianKernel(ksize=size, sigma=sigma)
    g_kernel = np.outer(g.T, g)
    return convolve2d(img, g_kernel, mode='same', boundary='symm')

def dog_filter(img, size=7, sigma=1):
    g = cv2.getGaussianKernel(ksize=size, sigma=sigma)
    g_kernel = np.outer(g.T, g)
    # print(g_kernel)
    dogx_kernel = dx_filter(g_kernel)
    dogy_kernel = dy_filter(g_kernel)
    # print(dogx_kernel, dogy_kernel)
    # print(dogx_kernel, dogy_kernel)
    return convolve2d(img, dogx_kernel, mode='same', boundary='symm'), convolve2d(img, dogy_kernel, mode='same', boundary='symm')
    

def part1_2():
    input_path = "./data/cameraman.png"
    in_path = "./output/part1/blur/input.png"
    gx_path = "./output/part1/blur/gx.png"
    gy_path = "./output/part1/blur/gy.png"
    gm_path = "./output/part1/blur/gm.png"
    gm_dog_path = "./output/part1/blur/gm_dog.png"
    bin_path = "./output/part1/blur/binarized.png"
    bin_dog_path = "./output/part1/blur/binarized_dog.png"

    img = cv2.imread(input_path, 0)
    img_blur = gaussian_filter(img)
    img_gx_ori = dx_filter(img)
    img_gx = dx_filter(img_blur)
    img_gy_ori = dy_filter(img)
    img_gy = dy_filter(img_blur)
    img_dogx, img_dogy = dog_filter(img_blur)
    img_gm_ori = np.sqrt(img_gx_ori**2 + img_gy_ori**2)
    img_gm = np.sqrt(img_gx**2 + img_gy**2)
    img_gm_dog = np.sqrt(img_dogx**2 + img_dogy**2)

    # Binarize the images
    img_bin_ori = binarize(img_gm_ori, 55)
    img_bin = binarize(img_gm, THRESHOLD)
    img_bin_dog = binarize(img_gm_dog, THRESHOLD)

    fig, (ax_blur, ax_mag, ax_bin) = plt.subplots(1, 3, figsize=(15, 6))
    ax_blur.imshow(img_blur, cmap='gray')
    ax_blur.set_title('Blurred')
    ax_blur.set_axis_off()

    ax_mag.imshow(img_gm, cmap='gray')
    ax_mag.set_title('Gradient magnitude')
    ax_mag.set_axis_off()

    ax_bin.imshow(img_bin, cmap='gray')
    ax_bin.set_title('Gradient binarize')
    ax_bin.set_axis_off()
    plt.savefig('./output/part1/blur/gradient_binarized.png')

    fig, (ax_ori, ax_blur) = plt.subplots(2, 5, figsize=(15, 9))
    ax_ori[0].imshow(img, cmap='gray')
    ax_ori[0].set_title('Original')
    ax_ori[0].set_axis_off()
    ax_ori[1].imshow(img_gx_ori, cmap='gray')
    ax_ori[1].set_title('Gradient X')
    ax_ori[1].set_axis_off()
    ax_ori[2].imshow(img_gy_ori, cmap='gray')
    ax_ori[2].set_title('Gradient Y')
    ax_ori[2].set_axis_off()
    ax_ori[3].imshow(img_gm_ori, cmap='gray')
    ax_ori[3].set_title('Gradient magnitude')
    ax_ori[3].set_axis_off()
    ax_ori[4].imshow(img_bin_ori, cmap='gray')
    ax_ori[4].set_title('Gradient binarize')
    ax_ori[4].set_axis_off()
    ax_blur[0].imshow(img_blur, cmap='gray')
    ax_blur[0].set_title('Blurred')
    ax_blur[0].set_axis_off()
    ax_blur[1].imshow(img_gx, cmap='gray')
    ax_blur[1].set_title('Gradient X')
    ax_blur[1].set_axis_off()
    ax_blur[2].imshow(img_gy, cmap='gray')
    ax_blur[2].set_title('Gradient Y')
    ax_blur[2].set_axis_off()
    ax_blur[3].imshow(img_gm, cmap='gray')
    ax_blur[3].set_title('Gradient magnitude')
    ax_blur[3].set_axis_off()
    ax_blur[4].imshow(img_bin, cmap='gray')
    ax_blur[4].set_title('Gradient binarize')
    ax_blur[4].set_axis_off()
    plt.tight_layout()
    plt.savefig('./output/part1/blur/compare1.png')

    fig, (ax, ax_dog) = plt.subplots(2, 4, figsize=(20, 8))
    ax[0].imshow(img_gx, cmap='gray')
    ax[0].set_title('Gradient X', fontsize=18)
    ax[0].set_axis_off()
    ax[1].imshow(img_gy, cmap='gray')
    ax[1].set_title('Gradient Y', fontsize=18)
    ax[1].set_axis_off()
    ax[2].imshow(img_gm, cmap='gray')
    ax[2].set_title('Gradient magnitude', fontsize=18)
    ax[2].set_axis_off()
    ax[3].imshow(img_bin, cmap='gray')
    ax[3].set_title('Gradient binarize', fontsize=18)
    ax[3].set_axis_off()
    ax_dog[0].imshow(img_dogx, cmap='gray')
    ax_dog[0].set_axis_off()
    ax_dog[1].imshow(img_dogy, cmap='gray')
    ax_dog[1].set_axis_off()
    ax_dog[2].imshow(img_gm_dog, cmap='gray')
    ax_dog[2].set_axis_off()
    ax_dog[3].imshow(img_bin_dog, cmap='gray')
    ax_dog[3].set_axis_off()
    fig.text(0.02, 0.75, 'Original', va='center', rotation='vertical', fontsize=18)
    fig.text(0.02, 0.25, 'DoG Filter', va='center', rotation='vertical', fontsize=18)
    plt.tight_layout()
    plt.savefig('./output/part1/blur/compare2.png')

    cv2.imwrite(in_path, img_blur)
    cv2.imwrite(gx_path, img_gx)
    cv2.imwrite(gy_path, img_gy)
    cv2.imwrite(gm_path, img_gm)
    cv2.imwrite(gm_dog_path, img_gm_dog)
    cv2.imwrite(bin_path, img_bin)
    cv2.imwrite(bin_dog_path, img_bin_dog)

if __name__ == "__main__":
    # part1_1()
    part1_2()