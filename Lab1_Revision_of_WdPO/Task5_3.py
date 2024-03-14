import cv2 as cv
import numpy as np

def main():
    # Load the image
    img = cv.imread('Pictures/coins.jpg')

    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv.GaussianBlur(gray, (15, 15), 0)

    # Apply Hough Circle Transform
    circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=50, maxRadius=100)
    circles = np.uint16(np.around(circles))

    # Draw the circles on the image
    for i in circles[0,:]:
        cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

    # Define coin values and radius thresholds
    coin_values = [0.1, 1]
    radius_thresholds = [75, 100]

    # Calculate total value
    total_value = 0
    for i in circles[0,:]:
        radius = i[2]
        for j in range(len(radius_thresholds)):
            if radius < radius_thresholds[j]:
                total_value += coin_values[j]
                break

    print(f'Total value of coins: {total_value:.2f}')

    # Display the image with circles
    cv.imshow('Detected Circles', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()