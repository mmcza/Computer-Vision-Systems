import cv2 as cv
import numpy as np
import os

def empty_callback(value):
    pass

def main():
    img = cv.imread("Pictures/dog1.jpg")
    cv.namedWindow("Image 1")
    rows, columns = img.shape[:2]

    img1_gray = cv.cvtColor(img[:int(rows/2), :int(columns/2)], cv.COLOR_BGR2GRAY)
    img1_red = img[:int(rows / 2), int(columns / 2):, 2]
    img1_green = img[int(rows / 2):, :int(columns / 2), 1]
    img1_blue = img[int(rows / 2):, int(columns / 2):, 0]

    img1_top = np.concatenate((img1_gray, img1_red), axis=1)
    img1_bot = np.concatenate((img1_green, img1_blue), axis=1)
    img1 = np.concatenate((img1_top,img1_bot), axis=0)

    cv.createTrackbar('R', 'Image 1', 0, 255, empty_callback)
    cv.createTrackbar('G', 'Image 1', 0, 255, empty_callback)
    cv.createTrackbar('B', 'Image 1', 0, 255, empty_callback)
    cv.createTrackbar('BW', 'Image 1', 0, 255, empty_callback)

    key = ord('a')
    while key != ord('q'):
        cv.imshow("Image 1", img1)

        # GET Trackbar Positions
        r_thresh = cv.getTrackbarPos('R', 'Image 1')
        g_thresh = cv.getTrackbarPos('G', 'Image 1')
        b_thresh = cv.getTrackbarPos('B', 'Image 1')
        bw_thresh = cv.getTrackbarPos('BW', 'Image 1')

        # Tresholding
        ret, thresh_r = cv.threshold(img1_red, r_thresh, 255, cv.THRESH_BINARY)
        ret, thresh_g = cv.threshold(img1_green, g_thresh, 255, cv.THRESH_BINARY)
        ret, thresh_b = cv.threshold(img1_blue, b_thresh, 255, cv.THRESH_BINARY)
        ret, thresh_bw = cv.threshold(img1_gray, bw_thresh, 255, cv.THRESH_BINARY)

        img2_top = np.concatenate((thresh_bw, thresh_r), axis=1)
        img2_bot = np.concatenate((thresh_g, thresh_b), axis=1)
        img2 = np.concatenate((img2_top, img2_bot), axis=0)

        cv.imshow("Image 2", img2)

        ### Image 3
        img3 = img.copy()
        img3[:int(rows/2), :int(columns/2)] = cv.cvtColor(thresh_bw, cv.COLOR_GRAY2BGR)
        img3[:int(rows / 2), int(columns / 2):, 2] = thresh_r
        img3[int(rows / 2):, :int(columns / 2), 1] = thresh_g
        img3[int(rows / 2):, int(columns / 2):, 0] = thresh_b

        cv.imshow("Image 3", img3)

        key = cv.waitKey(10)

if __name__ == '__main__':
    main()