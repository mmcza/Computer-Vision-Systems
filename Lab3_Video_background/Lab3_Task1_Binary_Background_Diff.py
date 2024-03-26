import cv2 as cv
import numpy as np

def nothing(x):
    pass
def main():
    # get image from camera
    cap = cv.VideoCapture(0)
    cv.namedWindow('Current Image')
    cv.namedWindow('Background')
    cv.namedWindow('Foreground')
    # create a trackbar in foreground window
    cv.createTrackbar('Threshold', 'Foreground', 0, 255, nothing)
    background_model = []
    current_frame = []

    while True:
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # get the threshold value from the trackbar
        threshold = cv.getTrackbarPos('Threshold', 'Foreground')

        # show the images
        cv.imshow('Current Image', gray)
        #cv2.imshow('Foreground', fgmask)
        # break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.waitKey(1) & 0xFF == ord('a'):
            background_model = gray
            cv.imshow('Background', background_model)
        # getting current frame when 'x' is pressed
        # if cv.waitKey(1) & 0xFF == ord('x') and len(background_model)>0:
        #     current_frame = gray
        #     fgmask = cv.absdiff(background_model, np.array(current_frame))
        #     ret, fgmask = cv.threshold(fgmask, threshold, 255, cv.THRESH_BINARY)
        #     cv.imshow('Foreground', fgmask)
        current_frame = gray
        if len(background_model)>0 and len(current_frame)>0:
            fgmask = cv.absdiff(background_model, np.array(current_frame))
            ret, fgmask = cv.threshold(fgmask, threshold, 255, cv.THRESH_BINARY)
            cv.imshow('Foreground', fgmask)


if __name__ == '__main__':
    main()

