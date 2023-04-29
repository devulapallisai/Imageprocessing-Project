# ----------- The below part of the code is written by Fasal Mohamed Roll No : EE20BTECH11016 -----------

import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.fftpack import dctn
from collections import defaultdict

# Define the parameters for the algorithm
b = 8  # block size
p = 0.5  # parameter for number of coefficients retained
q = 20  # number of coefficients to compare

# Convert DCT quantised matrix to vector in zig-zag fashion
def flatten(block):
    M, N = block.shape
    vector = []
    for k in range(M+N-1):
        indices = [(k-i, i) if k-i < M else (k-i, i+abs(M-N)) for i in range(min(k+1, N))]
        indices = indices if k % 2 == 0 else [(j, i) for i, j in indices]
        vector += [block[i, j] for i, j in indices if j < N and i < M]
    return vector

# Apply DCT to each block and keep only low-frequency coefficients
def vectorise(block, i, j, p):
    M, N = block.shape
    # print("In vectorise")
    Q = np.array([[16, 11, 10, 16,  24,  40,  51,  61],
                  [12, 12, 14, 19,  26,  58,  60,  55],
                  [14, 13, 16, 24,  40,  57,  69,  56],
                  [14, 17, 22, 29,  51,  87,  80,  62],
                  [18, 22, 37, 56,  68, 109, 103,  77],
                  [24, 35, 55, 64,  81, 104, 113,  92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103,  99]])
    dct_block = dctn(block)
    quantized_dct = np.round(dct_block / Q)    
    vector = flatten(quantized_dct)[:int(p * M * N)]
    return i, j, vector

# Divide the image into overlapping blocks and obtain lexicographically sorted vectors
def get_vectors(image, b, p):
    M, N = image.shape
    vectors = []
    for i in range(M - b + 1):
        for j in range(N - b + 1):
            # print("In get_vectors: ", i, " ", j)
            vectors.append(vectorise(image[i:i+b, j:j+b], i, j, p))

    vectors_sorted = sorted(vectors, key=lambda x: (tuple(x[2]), x[0], x[1]))
    return vectors_sorted

# Test neighboring rows for similarity and calculate shift vectors and obtain their frequency

# Ideal get_vectors() to be used; but too time-intensive
# def get_distances(vectors, b, q):
#     n = len(vectors)
#     distances = defaultdict(int)

#     for i, (i1, j1, vec1) in enumerate(vectors):
#         for j in range(i+1, n):
#             vec2 = vectors[j][2]
#             if not np.allclose(vec1[:q], vec2[:q]):
#                 break
#             else:
#                 # print("In get_distances: ", i, " ", j)
#                 i2, j2 = vectors[j][0], vectors[j][1]
#                 diff1 = np.abs(i1-i2)
#                 diff2 = np.abs(j1-j2)
#                 if diff1 >= b or diff2 >= b:
#                     distances[(i1-i2, j1-j2)] += 1
#                     distances[(i2-i1, j2-j1)] += 1
#     return distances

def get_distances(vectors, b, q):
    n = len(vectors)
    distances = defaultdict(int)

    for i, (i1, j1, vec1) in enumerate(vectors):
        for j in range(i+1, min(i+40, n)):
            vec2 = vectors[j][2]
            if not np.allclose(vec1[:q], vec2[:q]):
                break
            else:
                # print("In get_distances: ", i, " ", j)
                i2, j2 = vectors[j][0], vectors[j][1]
                diff1 = np.abs(i1-i2)
                diff2 = np.abs(j1-j2)
                if diff1 >= b or diff2 >= b:
                    distances[(i1-i2, j1-j2)] += 1
                    distances[(i2-i1, j2-j1)] += 1
    return distances

# Set the threshold frequency 
def mark_duplicates(image, shifts, b):
    print("In mark_duplicates")
    threshold = b*b
    for shift, count in shifts.items():
        if count > threshold:
            return 1
    return 0

# Display the output image
def detect_copy_move(path, b=8, p=0.1, q=2):
    image = plt.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Image converted\n\n\n")
    vectors = get_vectors(gray_image, b, p)
    print("Vectors obtained\n\n\n")
    shifts = get_distances(vectors, b, q)
    print("Distances obtained\n\n\n")
    result = mark_duplicates(image, shifts, b)
    return result

# def detect_copy_move(image, b=8, p=0.1, q=2):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     print("Image converted\n\n\n")
#     vectors = get_vectors(gray_image, b, p)
#     print("Vectors obtained\n\n\n")
#     shifts = get_distances(vectors, b, q)
#     print("Distances obtained\n\n\n")
#     result = mark_duplicates(image, shifts, b)
#     return result

paths = []
images = []
# path of the input folder of dataset which has all images

PATH = "./datasets/COFOMOD_v2/"
file_formats = [".png", ".jpg", ".jpeg"]  # image file formats

for x in os.listdir(PATH):
    istrue = np.array([x.endswith(file_format) for file_format in file_formats])
    if np.any(istrue):
        # detecting png,jpg and jpeg files which are most common image files (in dataset it has only these filetypes)
        paths.append(PATH+x)
        images.append(x)

# computing statistics like accuracy, precision, recall, f1_score to show robustness of our approach to solve the problem

false_positive = 0
false_negative = 0
true_negative = 0
true_positive = 0

# image = plt.imread(PATH + '002_O.png')
# print(detect_copy_move(image))

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
    stre = images[index].split("_")
    if (len(stre[1]) != 1):
        # we are skipping bit mask images as they are not images that we have to consider
        x = x-1
    else:
        detected = detect_copy_move(path)
        # print(stre)
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

        print(f"Time elapsed for a image {t1-t2}")

    index = index+1


accuracy = 1 - (error/L)
precision = true_positive/(true_positive+false_positive)
recall = true_positive/(true_positive+false_negative)
f1_score = 2/((1/precision)+(1/recall))

print(f"Accuracy is {accuracy}")
print(f"Precision is {precision}")
print(f"Recall is {recall}")
print(f"F1 score is {f1_score}")
print(f"Time taken is {t1-t0}")