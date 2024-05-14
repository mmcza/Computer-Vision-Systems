import cv2 as cv
import numpy as np

def task1():
    # Load the image
    img = cv.imread("zad_dopasowanie_obiekty_do_znalezienia_rotated.jpg", cv.IMREAD_GRAYSCALE)
    # downscale the image
    img = cv.resize(img, (0, 0), fx=0.25, fy=0.25)
    # let the user select the object to find
    r = cv.selectROI(img)
    # Crop the object
    object = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    print(r)

    # detect keypoints using FAST
    fast = cv.FastFeatureDetector_create()
    fast_keypoints = fast.detect(object)

    # calculate descriptors for the object
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints, descriptors = brief.compute(img, fast_keypoints)

    # move the keypoints to the object
    for keypoint in keypoints:
        keypoint.pt = (keypoint.pt[0] + r[0], keypoint.pt[1] + r[1])

    # draw keypoints
    img_keypoints = cv.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
    cv.imshow("FAST keypoints", img_keypoints)

    # load the image to find the object in
    img_to_find = cv.imread("zad_dopasowanie_obiekty_wzorcowe_rotated.jpg", cv.IMREAD_GRAYSCALE)
    # downscale the image
    img_to_find = cv.resize(img_to_find, (0, 0), fx=0.25, fy=0.25)

    # detect keypoints using FAST
    fast_keypoints_to_find = fast.detect(img_to_find)

    # calculate descriptors for the image
    keypoints_to_find, descriptors_to_find = brief.compute(img_to_find, fast_keypoints_to_find)

    # match the descriptors
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors, descriptors_to_find)

    # draw the matches
    img_matches = cv.drawMatches(img, keypoints, img_to_find, keypoints_to_find, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches = cv.resize(img_matches, (0, 0), fx=0.5, fy=0.5)
    cv.imshow("Matches", img_matches)

    # Only include points that have a match
    src_pts = np.float32([keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_to_find[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Use RASNAC to find the object
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    warped_img = cv.warpPerspective(img, H, (img_to_find.shape[1], img_to_find.shape[0]))
    cv.imshow("Warped image", warped_img)

    cv.waitKey(0)

def task2():
    # Load the image
    img = cv.imread("zdjecie_1.jpg")
    # downscale the image
    img = cv.resize(img, (0, 0), fx=0.25, fy=0.25)

    # detect keypoints using FAST
    fast = cv.FastFeatureDetector_create()
    fast_keypoints = fast.detect(img)

    # calculate descriptors for the object
    orb = cv.ORB_create()
    keypoints, descriptors = orb.compute(img, fast_keypoints)

    # draw keypoints
    img_keypoints = cv.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
    cv.imshow("FAST keypoints", img_keypoints)

    # load the image to find the object in
    img_to_find = cv.imread("zdjecie_2.jpg")
    # downscale the image
    img_to_find = cv.resize(img_to_find, (0, 0), fx=0.25, fy=0.25)

    # detect keypoints using FAST
    fast_keypoints_to_find = fast.detect(img_to_find)

    # calculate descriptors for the image
    keypoints_to_find, descriptors_to_find = orb.compute(img_to_find, fast_keypoints_to_find)

    # match the descriptors
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors, descriptors_to_find)

    # draw the matches
    img_matches = cv.drawMatches(img, keypoints, img_to_find, keypoints_to_find, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches = cv.resize(img_matches, (0, 0), fx=0.5, fy=0.5)
    cv.imshow("Matches", img_matches)

    # Only include points that have a match
    src_pts = np.float32([keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_to_find[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Use RASNAC to find the transformation
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    warped_img = cv.warpPerspective(img, H, (img_to_find.shape[1], img_to_find.shape[0]))

    # stitch the images together
    panorama = np.maximum(warped_img, img_to_find)

    cv.imshow("Warped image", warped_img)
    cv.imshow("Panorama", panorama)

    cv.waitKey(0)

if __name__ == "__main__":
    #task1()
    task2()