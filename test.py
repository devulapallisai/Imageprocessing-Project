import cv2
import os
import numpy as np


def detect_copy_move(image_path, block_size=8, ksize=9, sigma=5, theta=1*np.pi/4, lambd=10, gamma=0.5):
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Divide the image into blocks
    blocks = []
    for i in range(0, gray.shape[0], block_size):
        for j in range(0, gray.shape[1], block_size):
            block = gray[i:i+block_size, j:j+block_size]
            blocks.append(block)

    # Define the Gabor filter bank
    filters = []

    for i in range(4):
        for j in range(4):
            k = ksize
            sigma = sigma
            theta = j * np.pi / 4
            lambd = lambd
            gamma = gamma
            gabor_filter = cv2.getGaborKernel(
                (k, k), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
            filters.append(gabor_filter)

    # Apply the Gabor filters to each block
    block_features = []
    for block in blocks:
        features = []
        for gabor_filter in filters:
            filtered = cv2.filter2D(block, cv2.CV_8UC3, gabor_filter)
            features.append(np.mean(filtered))
        block_features.append(features)

    # Compare each block with all other blocks
    for i in range(len(blocks)):
        for j in range(len(blocks)):
            if i == j:
                continue
            diff = np.array(block_features[i]) - np.array(block_features[j])
            distance = np.linalg.norm(diff)
            if distance < 3:
                # Found a copy-move forgery
                return True

    # No copy-move forgery found
    return False


paths = []
PATH = "datasets/COFOMOD_v2/"

for x in os.listdir(PATH):
    if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg"):
        # Prints only text file present in My Folder
        paths.append(PATH+x)

false_positive = 0
false_negative = 0
true_negative = 0
true_positive = 0

x = 0
error = 0
L = 100

for path in paths:
    if (x == L):
        '''For L images in this folder'''
        break
    x += 1
    detection = detect_copy_move(path)
    stre = path.split("_")
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

    # print(detection, real_detected)

accuracy = error/L
precision = true_positive/(true_positive+false_positive)
recall = true_positive/(true_positive+false_negative)
f1_score = 2/((1/precision)+(1/recall))

print(f"Accuracy is {accuracy}")
print(f"Precision is {precision}")
print(f"Recall is {recall}")
print(f"F1 score is {f1_score}")
