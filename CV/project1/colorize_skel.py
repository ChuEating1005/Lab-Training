# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import cv2
import skimage as sk
from skimage import io, color
from skimage.metrics import structural_similarity as SSIM
from skimage.transform import rescale
import os 

# some parameters
MAX_DISPLACEMENT_LOWRES = 30
MAX_DISPLACEMENT_HIGHRES = 40
NUM_SCALE = 4
MATCHING_METRIC = "NCC"
SEARCH_METHOD = "PYR"
IMAGE_NAME = "monastery.jpg"

# some image matching metric
def metric(im1, im2):
    if MATCHING_METRIC == "SSD":
        return SSD(im1, im2)
    elif MATCHING_METRIC == "NCC":
        return NCC(im1, im2)
    elif MATCHING_METRIC == "SSIM":
        return SSIM(im1,im2)
def SSD(im1, im2):
    # sum of squared differences
    return np.sum((im1 - im2)**2)
def NCC(im1, im2):
    # normalized cross correlation
    return np.sum(np.multiply(im1 - np.mean(im1), im2 - np.mean(im2)) / (np.std(im1) * np.std(im2)))


def auto_crop(image):
    print(image.shape)
    if len(image) > 1000:
        # img = rescale(image, 0.25)
        img = rescale(image, 0.25, channel_axis=-1)
    else:
        img = image

    grayscale_image = color.rgb2gray(img)

    # Apply edge detection (Canny edge detection)
    image_blur = cv2.GaussianBlur(grayscale_image, (5,5), 0)
    edges = cv2.Canny((image_blur * 255).astype(np.uint8), threshold1=50, threshold2=20)

    cv2.imwrite("edge.png", edges)

    row_means = np.mean(edges / 255, axis=1)
    col_means = np.mean(edges / 255, axis=0)

    # Threshold to find edges (you can adjust the threshold value)
    row_threshold = 0.2
    col_threshold = 0.2

    # Find the innermost rows/columns that exceed the threshold
    rows_with_edges = np.where(row_means > row_threshold)[0]
    cols_with_edges = np.where(col_means > col_threshold)[0]
    # print(rows_with_edges)
    # print(cols_with_edges)
    row_negmax = np.where(rows_with_edges < 0.1 * len(row_means), rows_with_edges, -np.inf).max()
    row_posmin = np.where(rows_with_edges > 0.9 * len(row_means), rows_with_edges, np.inf).min()
    col_negmax = np.where(cols_with_edges < 0.1 * len(col_means), cols_with_edges, -np.inf).max()
    col_posmin = np.where(cols_with_edges > 0.9 * len(col_means), cols_with_edges, np.inf).min()
    # print(len(row_means), len(col_means))
    # print(row_negmax, row_posmin, col_negmax, col_posmin)
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

    print(top_row, bottom_row, left_col, right_col)
    if (len(image) < 1000):
        cropped_image = image[top_row:bottom_row+1, left_col:right_col+1]
    else:
        cropped_image = image[top_row*4:bottom_row*4+1, left_col*4:right_col*4+1]
    return edges, cropped_image
# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
def align(image, ref):
    if SEARCH_METHOD == "EX":
        ### Exhaustive search ###
        max_displacement = MAX_DISPLACEMENT_LOWRES
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
        scales = []
        if len(image) < 1000:
            scales = [1]
        else:
            scales = [2**(-i+1) for i in range(NUM_SCALE, 0, -1)]
        shift = (0, 0)
        for i in range(len(scales)):
            scale = scales[i]
            img_scale = sk.feature.canny(rescale(image, scale))
            ref_scale = sk.feature.canny(rescale(ref, scale))

            shift = (shift[0]*2, shift[1]*2)
            img_scale = np.roll(img_scale, shift, axis=(0, 1))

            # img_scale = img_scale[len(img_scale)//5:4*len(img_scale)//5, len(img_scale[0])//5:4*len(img_scale[0])//5]
            # ref_scale = ref_scale[len(ref_scale)//5:4*len(ref_scale)//5, len(ref_scale[0])//5:4*len(ref_scale[0])//5]
            
            max_score = float('-inf')
            new_shift = (0, 0)
            max_displacement = MAX_DISPLACEMENT_HIGHRES
            if i == 0:
                shift_list = [(i,j) for i in range(-max_displacement, max_displacement+1) for j in range(-max_displacement, max_displacement+1)]
            else:
                shift_list = [(i,j) for i in range(-3, 4) for j in range(-3, 4)]
            for i, j in shift_list:
                new_image = np.roll(img_scale, (i, j), axis=(0, 1))
                score = metric(new_image, ref_scale)
                if(score > max_score):
                    max_score = score
                    new_shift = (i, j)
            shift = (shift[0] + new_shift[0], shift[1] + new_shift[1])
        image = np.roll(image, shift, axis=(0, 1))
        print(shift)
        return image
        
if __name__ == "__main__":
    files = ['lady.tif']
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
        # io.imsave('b.png', b)
        # io.imsave('g.png', g)
        # io.imsave('r.png', r)
        

        ag = align(g, b)
        ar = align(r, b)
        # create a color image
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
        io.imsave(fname + '_origin.png', im_out)
        io.imsave(fname + '_edges.png', im_edges)
        io.imsave(fname + '_crop.png', im_crop)

    # display the image
    # io.imshow(im_out)
    # io.show()