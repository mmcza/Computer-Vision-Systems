import cv2 as cv
import numpy as np

def task1():
    img = cv.imread('tomatoes_and_apples.jpg')
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Define the kernel for morphological operations
    kernel = np.ones((15, 15), np.uint8)

    # Define the range of red color in HSV
    lower_red = np.array([0, 50, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv.inRange(img_hsv, lower_red, upper_red)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # sure background area
    sure_bg = cv.dilate(mask, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(mask, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    # create JET colormap
    markers = np.uint8(markers)
    markers = cv.applyColorMap(markers, cv.COLORMAP_JET)


    cv.imshow('Original Image', img)
    cv.imshow('Mask', mask)
    cv.imshow('Sure Background', sure_bg)
    cv.imshow('Sure Foreground', sure_fg)
    cv.imshow('Markers', markers)

    cv.waitKey(0)
    cv.destroyAllWindows()

def task2():
    img = cv.imread('cars.png')  # Read the image in color
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    # create JET colormap
    markers = np.uint8(markers)
    markers = cv.applyColorMap(markers, cv.COLORMAP_JET)

    cv.imshow('Original Image', img)
    cv.imshow('Threshold', thresh)
    cv.imshow('Sure Background', sure_bg)
    cv.imshow('Sure Foreground', sure_fg)
    cv.imshow('Markers', markers)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Global variables
drawing = False
ix, iy = -1, -1
img = None
rect_endpoint_tmp = []
rect_endpoint = []

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, rect_endpoint_tmp, rect_endpoint

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rect_endpoint_tmp = [(ix, iy), (x, y)]

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            img_temp = img.copy()
            x = min(max(0, x), img.shape[1] - 1)  # Ensure x is within the bounds of the image
            y = min(max(0, y), img.shape[0] - 1)  # Ensure y is within the bounds of the image
            cv.rectangle(img_temp, rect_endpoint_tmp[0], (x, y), (0, 255, 0), 1)
            cv.imshow('image', img_temp)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        x = min(max(0, x), img.shape[1] - 1)  # Ensure x is within the bounds of the image
        y = min(max(0, y), img.shape[0] - 1)  # Ensure y is within the bounds of the image
        rect_endpoint = [(ix, iy), (x, y)]
        cv.rectangle(img, rect_endpoint[0], rect_endpoint[1], (0, 255, 0), 1)

def task3():
    global img
    img = cv.imread('tumor.jpg')
    cv.namedWindow('image')
    cv.setMouseCallback('image', draw_rectangle)

    while(1):
        cv.imshow('image', img)
        if cv.waitKey(1) & 0xFF == 27:  # press ESC to exit
            break

    cv.destroyAllWindows()

    # Use the rectangle coordinates in the GrabCut function
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (rect_endpoint[0][0], rect_endpoint[0][1], rect_endpoint[1][0] - rect_endpoint[0][0], rect_endpoint[1][1] - rect_endpoint[0][1])
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)


if __name__ == '__main__':
    #task1()
    #task2()
    task3()