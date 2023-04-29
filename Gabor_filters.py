# This whole Code is written by Devulapalli Sai Prachodhan, Roll No : EE20BTECH11013
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt


def get_gabor_filters(size, sigma_min, sigma_max, K, n_theta, n_scale):
    ''' This function will precompute scaled and rotated versions of Gabor filters

    INPUT PARAMETERS :

    size      = Gabor filter square window size

    sigma     = sigma_min -> minimum standard deviation and sigma_max -> maximum standard deviation of filter

    n_theta   = number of angles that are into consideration

    K         = parameter to decide Uh and Ul as said in reference paper

    n_scale   = number of scales

    RETURNS :

    filters    = 2D matrix with generated Gabor filters where each row represents filters different rotation for
                 a given scale similar to SIFT(type = 2D array of 2D array)
    '''
    # Define the ranges of frequencies and orientations
    uh = K / (2 * np.pi * sigma_max)
    ul = K / (2 * np.pi * sigma_min)
    freqs = np.geomspace(ul, uh, n_scale)
    thetas = np.linspace(0, np.pi, n_theta, endpoint=False)

    # Generate the filters
    filters = []
    for freq in freqs:
        scale_filters = []
        for theta in thetas:
            # Generate the Gabor filter for the given frequency and orientation
            kernel = cv2.getGaborKernel(
                size,          # kernel size
                sigma_min,     # sigma (standard deviation)
                theta,         # orientation (radians)
                1.0/freq,      # wavelength
                1.0,           # aspect ratio
                0,            # phase offset
                cv2.CV_64F     # kernel datatype
            )
            scale_filters.append(kernel)
        rot_invariant = np.sum(scale_filters, axis=0)
        filters.append(rot_invariant)

    return filters


def generate_blocks(I, block_size=8):
    ''' This function will return sub blocks of image I

    INPUT PARAMETERS:

    I = grayscale image

    block_size = Image is divided into square blocks with each size of this variable for further processing
                 (type=integer)

    RETURNS:

    square overlapping blocks of size(block_size, block_size) from image I (2D array)

    '''
    blocks = []
    for i in range(0, len(I)-block_size+1):
        t = []
        for j in range(0, len(I[0])-block_size+1):
            block = I[i:i+block_size, j:j+block_size]
            t.append(block)  # (M-B+1)*(M-B+1) number of blocks
        blocks.append(t)
    return blocks


def feature_matrix_generation(blocks, filters):
    ''' This function will return feature matrix of whole image

    INPUT PARAMETERS:

    blocks = generated sub-blocks of image I

    filters = set of Gabor filters with rotated and scaled versions

    RETURNS:

    Square a row sorted matrix each representing feature vector of sub-block which has mean of all possible
    rotations for a given scale

    '''
    block_features = np.zeros(
        len(blocks)*len(blocks[0]), dtype=[('f_vector', np.ndarray, len(filters)),
                                           ('x', int), ('y', int)])
    p = 0
    for i in range(len(blocks)):
        for j in range(len(blocks[i])):
            filtered = np.array(
                [cv2.filter2D(blocks[i][j], cv2.CV_8UC3, filter) for filter in filters])
            feature_vector = np.mean(filtered, axis=(1, 2))
            # feature_vector has rotation invariant for given scale
            # storing feature vector of each block along with its top left corner coordinates
            block_features[p]["f_vector"] = feature_vector
            block_features[p]["x"] = i
            block_features[p]["y"] = j
            p = p+1
            # at each i,j position of Image I compute feature vector
    block_features = np.sort(block_features, order='f_vector')
    # lexographically sorting the above feature matrix
    return block_features


def detection(block_features_matrix, Nf=3, Nd=16, D=3):
    ''' This function will return whether image is an example of copy move forgery or not

    INPUT PARAMETERS:

    block_features_matrix = feature matrix A

    D = Euclidean Similarity Threshold

    Nf = Neighbourhood threshold

    Nd = Eucleadian distance Threshold

    RETURNS:

    0 or 1

    '''
    # Compare each block with all other blocks
    i = 0
    for block in block_features_matrix:
        # this will have all boolean matches between vectors i and j for all j<Nf+i
        k = np.minimum(i+Nf, len(block_features_matrix))
        for j in range(i, k, 1):
            # j-i < Nf
            if (np.linalg.norm(block["f_vector"]-block_features_matrix[j]["f_vector"]) < D):
                d = np.array([block["x"]-block_features_matrix[j]["x"],
                              block["y"]-block_features_matrix[j]["y"]])
                if (np.linalg.norm(d) > Nd):
                    return True
                    # similarity found i.e possible copy move
                else:
                    pass
                    # No similarity found i.e no copy move
            else:
                pass
        i = i+1
    return False


def detect_copy_move(image_path, filters, block_size=8, D=3, Nf=3, Nd=16):
    ''' This function will return whether image is an example of copy move forgery or not

    INPUT PARAMETERS:

    image_path = path for the image in the local directory that you are working so that
                 opencv can load the image(type: string)

    block_size = Image is divided into square blocks with each size of this variable for further processing
                 (type=integer)

    filters = Generated Gabor filters with different scales and rotation similar to SIFT
                 (type=array of 2D array where 2D array is of type complex)

    D = Euclidean Similarity Threshold

    Nf = Neighbourhood threshold

    Nd = Eucleadian distance Threshold

    RETURNS:

    True or False depending whether image is original or forged

    '''
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Divide the image into blocks
    blocks = generate_blocks(gray, block_size)

    # Apply the Gabor filters to each block and getting feature_matrix
    block_features_matrix = feature_matrix_generation(blocks, filters)

    return detection(block_features_matrix, Nf, Nd, D)


paths = []
images = []
# path of the input folder of dataset which has all images

PATH = "./datasets/COFOMOD_v2/"
file_formats = [".png", ".jpg", ".jpeg"]  # image file formats

# --------------------------GABOR FILTERS GENERATION ---------------------------------
filter_size = 16
no_filter_theta = 12  # theta angluar resolution = 30degree
filters = get_gabor_filters((filter_size, filter_size), sigma_min=5,
                            sigma_max=10, K=0.7, n_theta=no_filter_theta, n_scale=5)
# ------------------------------------------------------------------------------------

for x in os.listdir(PATH):
    istrue = np.array([x.endswith(file_format)
                       for file_format in file_formats])
    if np.any(istrue):
        # detecting png,jpg and jpeg files which are most common image files (in dataset it has only these filetypes)
        paths.append(PATH+x)
        images.append(x)

# computing statistics like accuracy, precision, recall, f1_score to show robustness of our approach to solve the problem

false_positive = 0
false_negative = 0
true_negative = 0
true_positive = 0

x = 0
error = 0
L = 40  # number of images from dataset to test upon
index = 0
t0 = time.time()
t1 = t0

for path in paths:
    t2 = t1
    if (x == L):
        '''For L images in this folder'''
        break
    x += 1
    # detection using pre computed Gabor filters
    stre = images[index].split("_")
    if (len(stre) == 2):
        # we are skipping bit mask images as they are not images that we have to consider
        if (stre[1] == "B.png" or stre[1] == "M.png"):
            # we are skipping bit mask images as they are not images that we have to consider
            x = x-1
        else:
            detected = detect_copy_move(path, filters)
            real_detected = 1 if stre[1] == "F" else 0

            if (real_detected == 0):
                if (detected == 0):
                    true_negative += 1
                else:
                    false_positive += 1
                    error += 1

            else:
                if (detected == 0):
                    false_negative += 1
                    error += 1
                else:
                    true_positive += 1

            t1 = time.time()

            print(f"{x} : Time elapsed for image is {t1-t2}")
    else:
        detected = detect_copy_move(path, filters)
        real_detected = 1 if stre[1] == "F" else 0

        if (real_detected == 0):
            if (detected == 0):
                true_negative += 1
            else:
                false_positive += 1
                error += 1

        else:
            if (detected == 0):
                false_negative += 1
                error += 1
            else:
                true_positive += 1

        t1 = time.time()

        print(f"{x} : Time elapsed for image is {t1-t2}")

    index = index+1

# Metrics considered

accuracy = 1 - (error/L)
precision = true_positive/(true_positive+false_positive)
recall = true_positive/(true_positive+false_negative)
f1_score = 2/((1/precision)+(1/recall))

print(f"Accuracy is {accuracy}")
print(f"Precision is {precision}")
print(f"Recall is {recall}")
print(f"F1 score is {f1_score}")
print(f"Time took is {t1-t0}")
