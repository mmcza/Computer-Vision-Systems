import cv2 as cv
import numpy as np
import os

def empty_callback(value):
    pass

def main():
    img = cv.imread("Pictures/super_important_document.jpg")
    cv.namedWindow("Image 1")
    height, width = img.shape[:2]
    ratio = height/1000
    new_width = int(width/ratio)
    rescaled_img = cv.resize(img, (new_width, 800))

    key = ord('a')
    while key != ord('q'):
        cv.imshow("Image 1", img)
        cv.imshow("Image 2", rescaled_img)

        key = cv.waitKey(10)

if __name__ == '__main__':
    main()