import cv2
import imutils
import numpy as np
import os

image_1_path = 'leaf_1.jpg'
image_2_path = 'leaf_2.jpg'

if not os.path.exists(image_1_path):
    print(f"File {image_1_path} does not exist.")
    exit()

if not os.path.exists(image_2_path):
    print(f"File {image_2_path} does not exist.")
    exit()

image_1 = cv2.imread(image_1_path)
image_2 = cv2.imread(image_2_path)

if image_1 is None:
    print(f"Failed to load {image_1_path}")
    exit()

if image_2 is None:
    print(f"Failed to load {image_2_path}")
    exit()

image_1 = cv2.resize(image_1, (300, 150))
image_2 = cv2.resize(image_2, (300, 150))

gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

difference = cv2.absdiff(gray_1, gray_2)
cv2.imshow('Difference', difference)

threshold = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

kernel = np.ones((5, 5), np.uint8)
dilate = cv2.dilate(threshold, kernel, iterations=2)
cv2.imshow('Dilation', dilate)

contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

for contour in contours:
    if cv2.contourArea(contour) > 100:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(image_2, (x, y), (x + w, y + h), (0, 0, 255), 2)

separator_height = image_1.shape[0] 
separator = np.zeros((separator_height, 10, 3), np.uint8)
result = np.hstack((image_1, separator, image_2))
cv2.imshow('Result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
