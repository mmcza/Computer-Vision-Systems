import cv2 as cv
import numpy as np

def main():
    # get image from camera
    cap = cv.VideoCapture(0)
    cv.namedWindow('Current Image')
    cv.namedWindow('Background')
    cv.namedWindow('Foreground')
    # create a trackbar in foreground window
    background_model = []
    current_frame = []

    while True:
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if len(background_model) == 0:
            background_model = gray
        cv.imshow('Background', background_model)

        # show the images
        cv.imshow('Current Image', gray)
        #cv2.imshow('Foreground', fgmask)
        # break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        current_frame = gray
        if len(background_model)>0 and len(current_frame)>0:
            # implementing approximate median background subtraction
            fgmask = background_model - current_frame
            # if certain value in fgmask is negative than substract 1 from background_model
            for i in range(len(fgmask)):
                for j in range(len(fgmask[i])):
                    if fgmask[i][j] < 0:
                        background_model[i][j] -= 1
                    if fgmask[i][j] > 0:
                        background_model[i][j] += 1
            cv.imshow('Foreground', fgmask)




if __name__ == '__main__':
    main()

