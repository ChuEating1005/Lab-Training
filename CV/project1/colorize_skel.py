import numpy as np
import cv2
import skimage as sk
from skimage import io, color
from skimage.metrics import structural_similarity as SSIM
from skimage.transform import rescale
import matplotlib.pyplot as plt
import os 

# some parameters
MAX_DISPLACEMENT_EX = 30
MAX_DISPLACEMENT_PYR = 40
NUM_SCALE = 4
MATCHING_METRIC = "NCC"
SEARCH_METHOD = "PYR"
IMAGE_NAME = "monastery.jpg"
THRESHOLD1 = 50
THRESHOLD2 = 20
ROW_THRESHOLD = 0.2
COL_THRESHOLD = 0.2

### Image matching metric ###
def metric(im1, im2):
    if MATCHING_METRIC == "SSD":
        # sum of squared differences
        return np.sum((im1 - im2)**2)
    
    elif MATCHING_METRIC == "NCC":
        # normalized cross correlation
        return np.sum(np.multiply(im1 - np.mean(im1), im2 - np.mean(im2)) / (np.std(im1) * np.std(im2)))
    
    elif MATCHING_METRIC == "SSIM":
        # structural similarity
        return SSIM(im1,im2)

### Edge Detection and Auto Cropping ###
def auto_crop(image):
    # Resize the image to make edge detection more precise
    if len(image) > 1000:
        img = rescale(image, 0.25, channel_axis=-1)
    else:
        img = image

    # Convert the image to grayscale
    grayscale_image = color.rgb2gray(img)

    # Apply edge detection (Canny edge detection)
    image_blur = cv2.GaussianBlur(grayscale_image, (5,5), 0)
    edges = cv2.Canny((image_blur * 255).astype(np.uint8), threshold1=THRESHOLD1, threshold2=THRESHOLD2)

    # Compute the mean of the rows and columns
    row_means = np.mean(edges / 255, axis=1)
    col_means = np.mean(edges / 255, axis=0)

    # Threshold to find edges
    row_threshold = ROW_THRESHOLD
    col_threshold = COL_THRESHOLD

    # Find the innermost rows/columns that exceed the threshold
    rows_with_edges = np.where(row_means > row_threshold)[0]
    cols_with_edges = np.where(col_means > col_threshold)[0]
    
    # Find the top row, bottom row, left column, and right column
    row_negmax = np.where(rows_with_edges < 0.1 * len(row_means), rows_with_edges, -np.inf).max()
    row_posmin = np.where(rows_with_edges > 0.9 * len(row_means), rows_with_edges, np.inf).min()
    col_negmax = np.where(cols_with_edges < 0.1 * len(col_means), cols_with_edges, -np.inf).max()
    col_posmin = np.where(cols_with_edges > 0.9 * len(col_means), cols_with_edges, np.inf).min()

    #  Check the detected edge is on the border
    if row_negmax == -np.inf:
        top_row = 0
    else:
        top_row = int(row_negmax)
    if row_posmin == np.inf:
        bottom_row = len(row_means)
    else:
        bottom_row = int(row_posmin)
    if col_negmax == -np.inf:
        left_col = 0
    else:
        left_col = int(col_negmax)
    if col_posmin == np.inf:
        right_col =  len(col_means)
    else:
        right_col = int(col_posmin)

    # Crop the image
    if (len(image) < 1000):
        cropped_image = image[top_row:bottom_row+1, left_col:right_col+1]
    else:
        cropped_image = image[top_row*4:bottom_row*4+1, left_col*4:right_col*4+1]

    return edges, cropped_image

### Image Alignment ###
def align(image, ref):
    if SEARCH_METHOD == "EX":
        ### Exhaustive search ###
        # Only search for a small range of displacement for low resolution images
        max_displacement = MAX_DISPLACEMENT_EX
        shift_list = [(i,j) for i in range(-max_displacement, max_displacement) for j in range(-max_displacement, max_displacement)]
        max_score = float('-inf')
        for i, j in shift_list:
            new_image = np.roll(image, (i, j), axis=(0, 1))
            score = metric(new_image, ref)
            if(score > max_score):
                max_score = score
                best_image = new_image
        return best_image
    else:
        ### Image Pyramid ###
        # Use to search for a large range of displacement for high resolution images
        # Faster than exhaustive search
        
        # Create a list of scales
        scales = []
        if len(image) < 1000:
            scales = [1]
        else:
            scales = [2**(-i+1) for i in range(NUM_SCALE, 0, -1)]

        shift = (0, 0)
        # For each scale, search for the best displacement
        for i in range(len(scales)):
            scale = scales[i]

            # Apply Canny edge detection to the image and the reference
            # Make the images that are different in 3 color channels look similar
            img_scale = sk.feature.canny(rescale(image, scale))
            ref_scale = sk.feature.canny(rescale(ref, scale))

            # Roll the image to the previous shift
            shift = (shift[0]*2, shift[1]*2)
            img_scale = np.roll(img_scale, shift, axis=(0, 1))
            max_score = float('-inf')
            new_shift = (0, 0)
            max_displacement = MAX_DISPLACEMENT_PYR

            # Create a list of displacements to search
            if i == 0:
                # For the lowest scale, search for a large range of displacement
                shift_list = [(i,j) for i in range(-max_displacement, max_displacement+1) for j in range(-max_displacement, max_displacement+1)]
            else:
                # For other scales, search for a smaller range of displacement
                shift_list = [(i,j) for i in range(-3, 4) for j in range(-3, 4)]

            # Search for the best displacement
            for i, j in shift_list:
                new_image = np.roll(img_scale, (i, j), axis=(0, 1))
                score = metric(new_image, ref_scale)
                if(score > max_score):
                    max_score = score
                    new_shift = (i, j)
            shift = (shift[0] + new_shift[0], shift[1] + new_shift[1])

        # Apply the best displacement to the original image
        image = np.roll(image, shift, axis=(0, 1))
        print(shift)
        return image

### Main Function ###
def main():
    files = ['emir.tif']
    # for file in os.listdir('data'):
    for file in files:
        print(file)
        # name of the input file
        imname = "./data/" + file

        # read in the image
        im = io.imread(imname)

        # compute the height of each part (just 1/3 of total)
        height = im.shape[0] // 3

        # separate color channels
        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        be = sk.feature.canny(b)
        ge = sk.feature.canny(g)
        re = sk.feature.canny(r)

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        ax[0][0].imshow(b, cmap='gray')
        ax[0][0].set_title('Blue Channel')
        ax[0][0].axis('off')    
        ax[0][1].imshow(g, cmap='gray')
        ax[0][1].set_title('Green Channel')
        ax[0][1].axis('off')
        ax[0][2].imshow(r, cmap='gray')
        ax[0][2].set_title('Red Channel')
        ax[0][2].axis('off')
        ax[1][0].imshow(be, cmap='gray')
        ax[1][0].axis('off')
        ax[1][1].imshow(ge, cmap='gray')
        ax[1][1].axis('off')
        ax[1][2].imshow(re, cmap='gray')
        ax[1][2].axis('off')
        plt.tight_layout()
        
        # align the imagess in the green and red channels to the blue channel
        ag = align(g, b)
        ar = align(r, b)
        
        # create color image, edges image, and cropped image
        im_out = np.dstack((ar, ag, b))
        im_edges, im_crop = auto_crop(im_out)
        im_out = sk.img_as_ubyte(im_out)
        im_edges = sk.img_as_ubyte(im_edges)
        im_crop = sk.img_as_ubyte(im_crop)

        # save the image
        root = os.path.splitext(file)[0]
        if not os.path.exists("./output/" + root): 
            os.makedirs("./output/" + root) 
        fname = './output/' + root + '/' + SEARCH_METHOD + '_' + MATCHING_METRIC 
        plt.savefig('./output/' + root + '/channels_compare.png')
        io.imsave(fname + '_origin.png', im_out)
        io.imsave(fname + '_edges.png', im_edges)
        io.imsave(fname + '_crop.png', im_crop)
        

if __name__ == "__main__":
    main()