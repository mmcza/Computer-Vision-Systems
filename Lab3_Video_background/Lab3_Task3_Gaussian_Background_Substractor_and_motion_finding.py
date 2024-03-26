from __future__ import print_function
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
 OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(0)
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

while True:
    ret, frame = capture.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if frame is None:
        break

    fgMask = backSub.apply(frame)
    ret, fgMask = cv.threshold(fgMask, 70, 255, cv.THRESH_BINARY)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)))
    white_pixels = cv.countNonZero(fgMask)
    if white_pixels > 0.05 * fgMask.size:
        cv.putText(frame, 'Motion Detected', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

    # add a frame around changed pixels
    contours, hierarchy = cv.findContours(fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break