import cv2 as cv
def search_contours(img, mask, col):
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)
    contour_number = 0
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 1000:
            new_img = cv.drawContours(img, [contour], -1, col, 2)

    return new_img
def main():
    img = cv.imread('Pictures/fruit.jpg')
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # define the range of colors for apples in HSV
    lower_apple = (20, 20, 90)
    upper_apple = (41, 255, 255)
    # create a mask for apples
    apple_mask = cv.inRange(hsv_img, lower_apple, upper_apple)
    # cv.imshow('Mask', apple_mask)
    # define the range of colors for oranges in HSV
    lower_oranges = (2, 20, 90)
    upper_oranges = (19, 255, 255)
    # create a mask for oranges
    orange_mask = cv.inRange(hsv_img, lower_oranges, upper_oranges)
    # cv.imshow('Mask', orange_mask)
    # search for contours
    new_img = search_contours(img, apple_mask, (0, 0, 255))
    new_img = search_contours(new_img, orange_mask, (255, 0 , 0))
    # display the image with detected contours
    cv.imshow('Contours', new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()