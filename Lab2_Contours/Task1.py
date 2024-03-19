import cv2 as cv
import numpy as np
def main():
    img = cv.imread('not_bad.jpg')
    #resizing image to get half of the original size
    img_resized = cv.resize(img, (0, 0), fx=0.3, fy=0.3)
    cv.imshow('Image', img_resized)
    # add tresholding to the image
    ret, thresh = cv.threshold(cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY), 50, 255, cv.THRESH_BINARY)
    # dilate the thresholded image with 5x5 kernel and 2 iterations
    thresh = cv.dilate(thresh, np.ones((5, 5)), iterations=1)
    cv.imshow('Thresholded Image', thresh)

    # find contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # draw contours on the image
    img_contours = img_resized.copy()
    print(hierarchy)
    color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 255)]

    # calculate center of mass for each contour
    points = []
    for i, contour in enumerate(contours[1:]):
        cv.drawContours(img_contours, contour, -1, color_list[i], 2)
        M = cv.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv.circle(img_contours, (cx, cy), 5, (0, 0, 255), -1)
            points.append([cx, cy])
    cv.imshow('Contours', img_contours)

    # calculate height and width
    width_l = np.sqrt(((points[0][0] - points[1][0]) ** 2) + ((points[0][1] - points[1][1]) ** 2))
    width_t = np.sqrt(((points[2][0] - points[3][0]) ** 2) + ((points[2][1] - points[3][1]) ** 2))
    width = max(int(width_t), int(width_l))

    height_r = np.sqrt(((points[0][0] - points[2][0]) ** 2) + ((points[0][1] - points[2][1]) ** 2))
    height_l = np.sqrt(((points[1][0] - points[3][0]) ** 2) + ((points[1][1] - points[3][1]) ** 2))
    height = max(int(height_r), int(height_l))

    # calculate the perspective transform
    input_pts = np.float32([points[3], points[1], points[0], points[2]])
    output_pts = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]])

    #print(input_pts)

    M = cv.getPerspectiveTransform(input_pts, output_pts)

    # apply the perspective transform
    final_image = cv.warpPerspective(img_resized, M, (width, height))

    cv.imshow('Final Image', final_image)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()