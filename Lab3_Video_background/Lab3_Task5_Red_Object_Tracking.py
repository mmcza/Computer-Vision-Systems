import cv2 as cv
import numpy as np

def main():
    # get image from camera
    cap = cv.VideoCapture(0)
    cv.namedWindow('Current Image')
    prev_frame = []
    while True:
        ret, frame = cap.read()
        # convert the image to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # define range of red color in HSV
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        # Threshold the HSV image to get only red colors
        mask1 = cv.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv.inRange(hsv, lower_red2, upper_red2)
        mask = cv.bitwise_or(mask1, mask2)
        # Bitwise-AND mask and original image
        res = cv.bitwise_and(frame, frame, mask=mask)
        gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        if len(prev_frame) == 0:
            prev_frame = gray
        # calculate the difference between the current frame and the previous frame
        diff = cv.absdiff(prev_frame, gray)
        prev_frame = gray
        # threshold the image
        ret, thresh = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)
        fgMask = cv.morphologyEx(thresh, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
        white_pixels = cv.countNonZero(fgMask)
        if white_pixels > 0.05 * fgMask.size:
            cv.putText(frame, 'Motion Detected', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

        # add a frame around changed pixels
        contours, hierarchy = cv.findContours(fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv.imshow('Current Image', frame)
        cv.imshow('Red range threshold', thresh)
        cv.imshow('Difference', diff)
        # break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
