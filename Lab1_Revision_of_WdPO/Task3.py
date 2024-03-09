import cv2 as cv
import numpy as np
import os

def main():
    img = cv.imread("Pictures/super_important_document.jpg", cv.IMREAD_GRAYSCALE)
    cv.namedWindow("Image 1")
    height, width = img.shape[:2]
    ratio = height/1200
    new_width = int(width/ratio)
    rescaled_img = cv.resize(img, (new_width, 1200))

    # no need for filters as they make it impossible to read anything
    median_img = cv.medianBlur(rescaled_img, 3)
    blurred_img = cv.blur(rescaled_img, (3, 3))
    gaussian_blur_img = cv.GaussianBlur(rescaled_img, (3, 3), 0)

    # threshold
    ret, th_img = cv.threshold(rescaled_img, 100, 255, cv.THRESH_BINARY)

    # adaptive threshold - creates noise, so it makes no sense to use it
    ad_th_b_img = cv.adaptiveThreshold(rescaled_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 2)
    ad_th_g_img = cv.adaptiveThreshold(rescaled_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 2)

    # morphological operations - after checking the results, in this situation it's better to not use any
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    erosion = cv.erode(th_img, kernel, iterations = 1)
    dilation = cv.dilate(th_img, kernel, iterations = 1)
    opening = cv.morphologyEx(th_img, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(th_img, cv.MORPH_CLOSE, kernel)

    cv.imwrite("Pictures/super_important_document_edited.jpg", th_img)

    original_size = os.path.getsize("Pictures/super_important_document.jpg")
    new_size = os.path.getsize("Pictures/super_important_document_edited.jpg")

    print("Original size was: "+str(original_size)+" B\nNew size is: "+str(new_size)+" B")

    key = ord('a')
    while key != ord('q'):
        cv.imshow("Image 1", img)
        cv.imshow("Image 2", th_img)
        key = cv.waitKey(10)

if __name__ == '__main__':
    main()