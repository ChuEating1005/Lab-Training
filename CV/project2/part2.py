from scipy.signal import convolve2d, unit_impulse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io as io
import os

### Parameters ###
KSIZE = 7
SIGMA = 1
ALPHA = 3
HIGHPASS_KSIZE = 10
HIGHPASS_SIGMA = 8
LOWPASS_KSIZE = 15
LOWPASS_SIGMA = 10
HYBRID_RATIO = 0.5
IMAGE1 = "submarine.bmp"
IMAGE2 = "fish.bmp"
IMAGE3 = "apple.jpeg"
IMAGE4 = "orange.jpeg"
LEVELS = 4

### Part 2-1: Image Sharpening ###
def gaussian_filter(img, ksize=KSIZE, sigma=SIGMA, color=False):
    g = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    g_kernel = np.outer(g.T, g)
    if color:
        r, g, b = cv2.split(img)
        r_blur = convolve2d(r, g_kernel, mode='same', boundary='symm')
        g_blur = convolve2d(g, g_kernel, mode='same', boundary='symm')
        b_blur = convolve2d(b, g_kernel, mode='same', boundary='symm')
        return np.dstack((r_blur, g_blur, b_blur))
    else:
        return convolve2d(img, g_kernel, mode='same', boundary='symm')

def sharpening(img):
    img_blur = gaussian_filter(img, color=True)
    img_detail = ALPHA * (img - img_blur)
    img_sharpened = np.clip(img + img_detail, 0, 255)
    return img_sharpened, np.clip(img_detail, 0, 255)

def unsharp_masking(img, ksize=KSIZE, sigma=SIGMA):
    g = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    g_kernel = np.outer(g.T, g)
    unsharp_mask_kernel = (1 + ALPHA) * unit_impulse((ksize,ksize), idx='mid') - ALPHA * g_kernel
    r, g, b = cv2.split(img)
    r_sharpened = convolve2d(r, unsharp_mask_kernel, mode='same', boundary='symm')
    g_sharpened = convolve2d(g, unsharp_mask_kernel, mode='same', boundary='symm')
    b_sharpened = convolve2d(b, unsharp_mask_kernel, mode='same', boundary='symm')
    img_sharpened = np.dstack((r_sharpened, g_sharpened, b_sharpened))
    img_sharpened = np.clip(img_sharpened, 0, 255)
    return img_sharpened

def frequency_analysis(img_origin, img_low, img_high, img_sharpened):
    magnitude_spectrum = []
    img = [img_origin, img_low, img_high, img_sharpened]
    for i in range(4):
        # Convert to grayscale if the image is in RGB
        gray_image = cv2.cvtColor(img[i].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        # Compute the 2D Fourier Transform of the image
        f_transform = np.fft.fft2(gray_image)
        
        # Shift the zero frequency component to the center
        f_shift = np.fft.fftshift(f_transform)
    
        # Compute the magnitude spectrum and use logarithmic scaling
        magnitude_spectrum.append(np.log(np.abs(f_shift) + 1))  # Adding 1 to avoid log(0)

    # High & Low Frequency Image Comparison
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0][0].imshow(cv2.cvtColor(img[1].astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax[0][0].set_title('Spatial Domain', fontsize=18)
    ax[0][0].axis('off')
    ax[0][1].imshow(magnitude_spectrum[1], cmap='hot')
    ax[0][1].set_title('Frequency Domain', fontsize=18)
    ax[0][1].axis('off')
    ax[1][0].imshow(cv2.cvtColor(img[2].astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax[1][0].axis('off')
    ax[1][1].imshow(magnitude_spectrum[2], cmap='hot')
    ax[1][1].axis('off')
    fig.text(0.02, 0.75, 'Low Frequency', va='center', rotation='vertical', fontsize=18)
    fig.text(0.02, 0.25, 'High Frequency', va='center', rotation='vertical', fontsize=18)
    fig.tight_layout()
    plt.savefig("./output/part2/2-1/frequency_analysis1.png")

    # Original & Sharpened Image Comparison
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0][0].imshow(cv2.cvtColor(img[0].astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax[0][0].set_title('Spatial Domain', fontsize=18)
    ax[0][0].axis('off')
    ax[0][1].imshow(magnitude_spectrum[0], cmap='hot')
    ax[0][1].set_title('Frequency Domain', fontsize=18)
    ax[0][1].axis('off')
    ax[1][0].imshow(cv2.cvtColor(img[3].astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax[1][0].axis('off')
    ax[1][1].imshow(magnitude_spectrum[3], cmap='hot')
    ax[1][1].axis('off')
    fig.text(0.02, 0.75, 'Original', va='center', rotation='vertical', fontsize=18)
    fig.text(0.02, 0.25, 'Sharpened', va='center', rotation='vertical', fontsize=18)
    fig.tight_layout()
    plt.savefig("./output/part2/2-1/frequency_analysis2.png")

def part2_1():
    input_path = "./data/taj.jpg"
    in_path = "./output/part2/2-1/input.png"
    blur_path = "./output/part2/2-1/blur.png"
    sharpen_path = "./output/part2/2-1/sharpen.png"
    unsharp_mask_path = "./output/part2/2-1/unsharp_mask.png"
    detail_path = "./output/part2/2-1/detail.png"

    img = cv2.imread(input_path)
    img_blur = gaussian_filter(img, color=True)
    img_sharpened, img_detail = sharpening(img)
    img_unsharp_mask = unsharp_masking(img)

    # Frequency Analysis
    frequency_analysis(img, img_blur, img_detail, img_sharpened)

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(img_sharpened.astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax[1].set_title("Sharpened")
    ax[1].axis("off")
    ax[2].imshow(cv2.cvtColor(img_unsharp_mask.astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax[2].set_title("Unsharp Mask Filter")
    ax[2].axis("off")
    fig.tight_layout()
    plt.savefig("./output/part2/2-1/compare.png")

    cv2.imwrite(in_path, img)
    cv2.imwrite(blur_path, img_blur)
    cv2.imwrite(detail_path, img_detail)
    cv2.imwrite(sharpen_path, img_sharpened)
    cv2.imwrite(unsharp_mask_path, img_unsharp_mask)

### Part 2-2: Hybrid Images ###
def highpass_filter(img, color=True):
    return (img - gaussian_filter(img, ksize=HIGHPASS_KSIZE, sigma=HIGHPASS_SIGMA, color=color)) + 128

def lowpass_filter(img, color=True):
    return gaussian_filter(img, ksize=LOWPASS_KSIZE, sigma=LOWPASS_SIGMA, color=color)

def hybrid_images(img1, img2, color=True, ratio=0.5):
    img1_lowpass = lowpass_filter(img1, color=color)
    img2_highpass = highpass_filter(img2, color=color)
    img_hybrid = ratio * img1_lowpass + (1 - ratio) * img2_highpass
    return img1_lowpass, img2_highpass, img_hybrid

def part2_2():
    img1_path = "./data/" + IMAGE1
    img2_path = "./data/" + IMAGE2
    img1_name = os.path.splitext(IMAGE1)[0]
    img2_name = os.path.splitext(IMAGE2)[0]

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1_lowpass, img2_highpass, img_hybrid = hybrid_images(img1, img2, color=True, ratio=HYBRID_RATIO)

    img1_lowpass = img1_lowpass.astype(np.uint8)
    img2_highpass = img2_highpass.astype(np.uint8)
    img_hybrid = img_hybrid.astype(np.uint8)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(cv2.cvtColor(img1_lowpass, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Lowpass Image")
    ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(img2_highpass, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Highpass Image")
    ax[1].axis("off")
    ax[2].imshow(cv2.cvtColor(img_hybrid, cv2.COLOR_BGR2RGB))
    ax[2].set_title("Hybrid Image")
    ax[2].axis("off")
    plt.savefig("./output/part2/2-2/compare_" + img1_name + "_" + img2_name + ".png")

    cv2.imwrite("./output/part2/2-2/" + img1_name + "_lowpass.png", img1_lowpass)
    cv2.imwrite("./output/part2/2-2/" + img2_name + "_highpass.png", img2_highpass)
    cv2.imwrite("./output/part2/2-2/hybrid_" + img1_name + "_" + img2_name + ".png", img_hybrid)

### Part 2-3: Gaussian and Laplacian Stacks ###
def gaussian_stack(img, name, ksize, sigma):
    stack = [img]
    img_blur = img
    for i in range(LEVELS):
        img_blur = gaussian_filter(stack[i], color=True, ksize=ksize, sigma=sigma)
        stack.append(img_blur)
    
    fig, ax = plt.subplots(1, LEVELS+1, figsize=(20, 5))
    fig.suptitle("Gaussian Stack for " + name, fontsize=18)
    fig.tight_layout()
    for i in range(LEVELS+1):
        if name != "mask":
            ax[i].imshow(cv2.cvtColor(stack[i].astype(np.uint8), cv2.COLOR_BGR2RGB))
        else:
            ax[i].imshow(stack[i], cmap="gray")
        ax[i].set_title("Level " + str(i+1))
        ax[i].axis("off")
    plt.savefig("./output/part2/2-3/gaussian_stack_" + name + ".png")

    return stack

def laplacian_stack(gaussian_stack, name):
    stack = []
    for i in range(LEVELS):
        stack.append(gaussian_stack[i] - gaussian_stack[i+1])
    stack.append(gaussian_stack[-1])

    fig, ax = plt.subplots(1, LEVELS+1, figsize=(20, 5))
    fig.suptitle("Laplacian Stack for " + name, fontsize=18)
    fig.tight_layout()
    for i in range(LEVELS+1):
        ax[i].imshow(cv2.cvtColor(stack[i].astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax[i].set_title("Level " + str(i+1))
        ax[i].axis("off")
    plt.savefig("./output/part2/2-3/laplacian_stack_" + name + ".png")
    return stack

### Part 2-4: Multiresolution Blending ###
def image_blending(img1_stack, img2_stack, mask_stack):
    blended_stack = []
    for i in range(LEVELS+1):
        blended_stack.append(0.9 * img1_stack[i] * mask_stack[i] + 0.9 * img2_stack[i] * (1 - mask_stack[i]))
    
    fig, ax = plt.subplots(1, LEVELS+1, figsize=(20, 5))
    fig.suptitle("Blended Image Stack", fontsize=18)
    fig.tight_layout()
    for i in range(LEVELS+1):
        ax[i].imshow(cv2.cvtColor(blended_stack[i].astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax[i].set_title("Level " + str(i+1))
        ax[i].axis("off")
    plt.savefig("./output/part2/2-4/blendimg_stack.png")

    blend_img = np.sum(blended_stack, axis=0)
    return blend_img

def part2_4():
    img1_path = "./data/" + IMAGE3
    img2_path = "./data/" + IMAGE4
    img1_name = os.path.splitext(IMAGE3)[0]
    img2_name = os.path.splitext(IMAGE4)[0]

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1_gaussian_stack = gaussian_stack(img1, name=img1_name, ksize=7, sigma=2)
    img2_gaussian_stack = gaussian_stack(img2, name=img2_name, ksize=7, sigma=2)
    img1_laplacian_stack = laplacian_stack(img1_gaussian_stack, name=img1_name)
    img2_laplacian_stack = laplacian_stack(img2_gaussian_stack, name=img2_name)

    mask = np.zeros(img1.shape[0:2])
    mask[:, :mask.shape[1]//2] = 1
    mask = np.dstack((mask, mask, mask))
    mask_stack = gaussian_stack(mask, name="mask", ksize=25, sigma=10)

    blend_img = image_blending(img1_laplacian_stack, img2_laplacian_stack, mask_stack)
    blend_img = blend_img.astype(np.uint8)

    cv2.imwrite("./output/part2/2-4/blend_" + img1_name + "_" + img2_name + ".png", blend_img)


if __name__ == "__main__":
    part2_1()
    # part2_2()
    # part2_4()