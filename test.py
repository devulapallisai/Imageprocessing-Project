import cv2
import os
import numpy as np


def gabor_filters(ksize=5, sigma=5, theta=np.pi/10, lambd=10, gamma=1):
    ''' This function will precompute scaled and rotated versions of Gabor filters

    INPUT PARAMETERS :


    ksize      = Gabor filter square window size

    sigma      = standard deviation for the Gabor filter

    theta      = angle of rotation 

    lambd      = wavelength of filter (can be attributed to scale change)

    gamma      = gamma correction factor

    RETURNS :

    filters    = 2D matrix with generated Gabor filters where each row represents filters different rotation for
                 a given scale similar to SIFT(type = 2D array of 2D array)
    '''
    # Define the Gabor filter bank
    filters = []

    # number of angles for different rotations
    no_of_angles = int(np.round(2*(np.pi)/theta))

    for scale in np.arange(0.5, 1.5, 0.1):
        # scales changing from 0.5 to 1.5 with each step of size 0.1
        scale_filters = []
        for j in range(no_of_angles):
            lambd = lambd
            gamma = gamma
            gabor_filter = cv2.getGaborKernel(
                (ksize, ksize), sigma, j*theta, lambd*scale, gamma, 0, ktype=cv2.CV_32F)
            scale_filters.append(gabor_filter)
        filters.append(scale_filters)

    return filters


def generate_blocks(I, block_size=8):
    ''' This function will return sub blocks of image I

    INPUT PARAMETERS :

    I          = grayscale image 

    block_size = Image is divided into square blocks with each size of this variable for further processing
                 (type = integer)

    RETURNS :

    Square blocks of size (block_size,block_size) from image I

    '''
    blocks = []
    for i in range(0, len(I), block_size):
        for j in range(0, len(I[0]), block_size):
            block = I[i:i+block_size, j:j+block_size]
            blocks.append(block)
    return blocks


def feature_matrix_generation(blocks, filters):
    ''' This function will return feature matrix of whole image 

    INPUT PARAMETERS :

    blocks  = generated sub-blocks of image I 

    filters = set of Gabor filters with rotated and scaled versions

    RETURNS :

    Square a row sorted matrix each representing feature vector of sub-block which has mean of all possible rotations for a given scale

    '''
    block_features = []
    for block in blocks:
        feature_vector = []
        for gabor_filters_rot in filters:
            filtered = []
            for filter in gabor_filters_rot:
                filtered_single = cv2.filter2D(block, cv2.CV_8UC3, filter)
                filtered.append(filtered_single)
            feature_vector.append(np.mean(filtered))
            # feature_vector has rotation invariant for given scale
        # sorting feature vector for each block
        feature_vector = np.sort(feature_vector)
        block_features.append(feature_vector)

    return block_features


def euclidean_distance(A, B):
    '''Returns euclidean similarity measure for vectors A and B

    INPUT PARAMETERS :

    A and B = vector of integers 

    RETURNS :

    euclidean distance also called l2-norm of A and B

    '''

    return np.linalg.norm(A-B)


def detect_copy_move(image_path, filters, block_size=8, D=3, Nf=3, Nd=16):
    ''' This function will return whether image is an example of copy move forgery or not

    INPUT PARAMETERS :

    image_path = path for the image in the local directory that you are working so that 
                 opencv can load the image  (type : string)

    block_size = Image is divided into square blocks with each size of this variable for further processing
                 (type = integer)

    filters    = Generated Gabor filters with different scales and rotation similar to SIFT
                 (type = array of 2D array where 2D array is of type complex)

    D          = Euclidean Similarity Threshold 

    Nf         = Neighbourhood threshold 

    Nd         = Eucleadian distance Threshold

    RETURNS :

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

    # Compare each block with all other blocks
    for i in range(len(block_features_matrix)):
        # this will have all boolean matches between vectors i and j for all j<Nf+i
        truth_vector = []
        for j in range(0, i+Nf):
            '''j-i<Nf'''
            if (euclidean_distance(block_features_matrix[i], block_features_matrix[j]) < D):
                if (block_size*euclidean_distance(np.array([i, i]), np.array([j, j])) > Nd):
                    truth_vector.append(False)
                    # similarity found i.e possible copy move
                else:
                    truth_vector.append(True)
                    # No similarity found i.e no copy move
            else:
                truth_vector.append(True)
        truth_vector = np.array(truth_vector)
        if (truth_vector.any()):
            # for all blocks whether it is true or not
            return True
    return False


paths = []
images = []
# path of the input folder of dataset which has all images

PATH = "datasets/COFOMOD_v2/"
file_formats = [".png", ".jpg", ".jpeg"]  # image file formats

# --------------------------GABOR FILTERS GENERATION ---------------------------------
filter_size = 5
filter_standard_dev = 10
filter_theta = np.pi/30  # theta
filters = gabor_filters(filter_size, filter_standard_dev, filter_theta)
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
L = 100  # number of images from dataset to test upon
index = 0

for path in paths:
    if (x == L):
        '''For L images in this folder'''
        break
    x += 1
    # detection using pre computed Gabor filters
    detection = detect_copy_move(path, filters)
    stre = images[index].split("_")
    if (len(stre[1]) != 1):
        x = x-1
    else:
        real_detected = 1 if stre[1] == "F" else 0

        if (real_detected == 0):
            if (detection == 0):
                true_negative += 1
            else:
                false_positive += 1
                error += 1

        else:
            if (detection == 0):
                false_negative += 1
                error += 1
            else:
                true_positive += 1

    index = index+1

accuracy = error/L
precision = true_positive/(true_positive+false_positive)
recall = true_positive/(true_positive+false_negative)
f1_score = 2/((1/precision)+(1/recall))

print(f"Accuracy is {accuracy}")
print(f"Precision is {precision}")
print(f"Recall is {recall}")
print(f"F1 score is {f1_score}")
