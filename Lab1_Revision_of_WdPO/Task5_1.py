import cv2 as cv

def main():
    img = cv.imread('Pictures/drone_ship.jpg', cv.IMREAD_GRAYSCALE)

    # Apply Canny edge detection
    edges = cv.Canny(img, threshold1=350, threshold2=500)

    # Display the image with detected edges
    cv.imshow('Edges', edges)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()