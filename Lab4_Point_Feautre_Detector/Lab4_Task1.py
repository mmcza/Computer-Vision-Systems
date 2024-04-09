import cv2 as cv
import numpy as np
import datetime

def task1():
    # read image
    img = cv.imread("detekcja_deskrypcja_dopasowanie/forward-1.bmp")

    # detect keypoints using FAST
    start_fast = datetime.datetime.now()
    fast = cv.FastFeatureDetector_create()
    fast_keypoints = fast.detect(img)
    fast_time = datetime.datetime.now() - start_fast
    print("FAST time: ", fast_time.microseconds, " microseconds")

    # detect keypoints using ORB
    start_orb = datetime.datetime.now()
    orb = cv.ORB_create()
    orb_keypoints = orb.detect(img)
    orb_time = datetime.datetime.now() - start_orb
    print("ORB time: ", orb_time.microseconds, " microseconds\n")
    print("Number of FAST keypoints: ", len(fast_keypoints))
    print("Number of ORB keypoints: ", len(orb_keypoints))

    # draw keypoints
    fast_img = cv.drawKeypoints(img, fast_keypoints, None, color=(0, 255, 0))
    orb_img = cv.drawKeypoints(img, orb_keypoints, None, color=(0, 255, 0))

    # display image
    cv.imshow("FAST", fast_img)
    cv.imshow("ORB", orb_img)
    cv.waitKey(0)

def task2():
    # read image
    img = cv.imread("detekcja_deskrypcja_dopasowanie/forward-1.bmp")

    # detect keypoints using FAST
    fast = cv.FastFeatureDetector_create()
    fast_keypoints = fast.detect(img)

    # use BRIEF to compute descriptors
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints, descriptors = brief.compute(img, fast_keypoints)
    print("BRIEF descriptors: ")
    print(descriptors)

    # use ORB to compute descriptors
    orb = cv.ORB_create()
    keypoints, descriptors = orb.compute(img, fast_keypoints)
    print("ORB descriptors: ")
    print(descriptors)

    # draw keypoints
    fast_img = cv.drawKeypoints(img, fast_keypoints, None, color=(0, 255, 0))

    # display image
    cv.imshow("FAST", fast_img)
    cv.waitKey(0)

def task3():
    # read image
    img = cv.imread("detekcja_deskrypcja_dopasowanie/forward-1.bmp")
    img2 = cv.imread("detekcja_deskrypcja_dopasowanie/forward-2.bmp")

    # detect keypoints using FAST
    fast = cv.FastFeatureDetector_create()
    fast_keypoints = fast.detect(img)
    fast_keypoints2 = fast.detect(img2)

    # detect keypoints using ORB
    orb = cv.ORB_create()
    orb_keypoints = orb.detect(img)
    orb_keypoints2 = orb.detect(img2)

    # use BRIEF to compute descriptors
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints, descriptors = brief.compute(img, fast_keypoints)
    keypoints2, descriptors2 = brief.compute(img2, fast_keypoints2)

    # use ORB to compute descriptors
    orb_keypoints, orb_descriptors = orb.compute(img, orb_keypoints)
    orb_keypoints2, orb_descriptors2 = orb.compute(img2, orb_keypoints2)

    # match descriptors
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors, descriptors2)
    # for i in range(20):
    #     print(matches[i].imgIdx, matches[i].queryIdx, matches[i].trainIdx, matches[i].distance)
    orb_matches = bf.match(orb_descriptors, orb_descriptors2)
    print("Number of matches from FAST algorithm and BRIEF descriptor: ", len(matches))
    print("Number of matches from ORB algorithm and ORB descriptor: ", len(orb_matches))

    # draw matching keypoints
    match_img = cv.drawMatches(img, fast_keypoints, img2, fast_keypoints2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_img_orb = cv.drawMatches(img, orb_keypoints, img2, orb_keypoints2, orb_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("Matching points from FAST algorithm and BRIEF descriptor", match_img)
    cv.imshow("Matching points from ORB algorithm and ORB descriptor", match_img_orb)
    cv.waitKey(0)

def task4():
    img = cv.imread("detekcja_deskrypcja_dopasowanie/rotate-1.bmp")
    img2 = cv.imread("detekcja_deskrypcja_dopasowanie/rotate-8.bmp")

    # detect keypoints using FAST
    fast = cv.FastFeatureDetector_create()
    fast_keypoints = fast.detect(img)
    fast_keypoints2 = fast.detect(img2)

    # detect keypoints using ORB
    orb = cv.ORB_create()
    orb_keypoints = orb.detect(img)
    orb_keypoints2 = orb.detect(img2)

    # use BRIEF to compute descriptors
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints, descriptors = brief.compute(img, fast_keypoints)
    keypoints2, descriptors2 = brief.compute(img2, fast_keypoints2)

    # use ORB to compute descriptors
    orb_keypoints, orb_descriptors = orb.compute(img, orb_keypoints)
    orb_keypoints2, orb_descriptors2 = orb.compute(img2, orb_keypoints2)

    # match descriptors
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors, descriptors2)
    # for i in range(20):
    #     print(matches[i].imgIdx, matches[i].queryIdx, matches[i].trainIdx, matches[i].distance)
    orb_matches = bf.match(orb_descriptors, orb_descriptors2)
    print("Number of matches from FAST algorithm and BRIEF descriptor: ", len(matches))
    print("Number of matches from ORB algorithm and ORB descriptor: ", len(orb_matches))

    # draw matching keypoints
    match_img = cv.drawMatches(img, fast_keypoints, img2, fast_keypoints2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_img_orb = cv.drawMatches(img, orb_keypoints, img2, orb_keypoints2, orb_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("Matching points from FAST algorithm and BRIEF descriptor", match_img)
    cv.imshow("Matching points from ORB algorithm and ORB descriptor", match_img_orb)
    cv.waitKey(0)

def task5():
    img = cv.imread("detekcja_deskrypcja_dopasowanie/perspective-1.bmp")
    img2 = cv.imread("detekcja_deskrypcja_dopasowanie/perspective-4.bmp")

    # detect keypoints using FAST
    fast = cv.FastFeatureDetector_create()
    fast_keypoints = fast.detect(img)
    fast_keypoints2 = fast.detect(img2)

    # detect keypoints using ORB
    orb = cv.ORB_create()
    orb_keypoints = orb.detect(img)
    orb_keypoints2 = orb.detect(img2)

    # use BRIEF to compute descriptors
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints, descriptors = brief.compute(img, fast_keypoints)
    keypoints2, descriptors2 = brief.compute(img2, fast_keypoints2)

    # use ORB to compute descriptors
    orb_keypoints, orb_descriptors = orb.compute(img, orb_keypoints)
    orb_keypoints2, orb_descriptors2 = orb.compute(img2, orb_keypoints2)

    # match descriptors
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors, descriptors2)
    # for i in range(20):
    #     print(matches[i].imgIdx, matches[i].queryIdx, matches[i].trainIdx, matches[i].distance)
    orb_matches = bf.match(orb_descriptors, orb_descriptors2)
    print("Number of matches from FAST algorithm and BRIEF descriptor: ", len(matches))
    print("Number of matches from ORB algorithm and ORB descriptor: ", len(orb_matches))

    # draw matching keypoints
    match_img = cv.drawMatches(img, fast_keypoints, img2, fast_keypoints2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_img_orb = cv.drawMatches(img, orb_keypoints, img2, orb_keypoints2, orb_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("Matching points from FAST algorithm and BRIEF descriptor", match_img)
    cv.imshow("Matching points from ORB algorithm and ORB descriptor", match_img_orb)
    cv.waitKey(0)

if __name__ == "__main__":
    #task1()
    #task2()
    task3()
    #task4()
    #task5()