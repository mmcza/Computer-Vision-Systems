import cv2 as cv
import numpy as np

def rotate_image(img, rotation):
    # get the image size and it's center
    height, width = img.shape[:2]
    image_center = (width/2, height/2)

    # get the rotation matrix
    rotation_mat = cv.getRotationMatrix2D(image_center, rotation, 1)

    # calculate the dimensions of the rotated image
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # adjust the rotation matrix
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate the image and get scale factor
    rotated_img = cv.warpAffine(img, rotation_mat, (bound_w, bound_h))
    scale_factor = min(width/bound_w, height/bound_h)
    resized_rotated_img = cv.resize(rotated_img, None, fx=scale_factor, fy=scale_factor)

    # create a black image with the original image size
    result = np.zeros_like(img)

    # get the dimensions of the resized rotated image and it's middle size
    rh, rw = resized_rotated_img.shape[:2]
    y = (height - rh) // 2
    x = (width - rw) // 2
    result[y:y+rh, x:x+rw] = resized_rotated_img

    return result

def main():
    img = cv.imread("Pictures/dog4.jpg")
    cv.namedWindow("Image")
    # adding trackbar to set rotation of the image
    cv.createTrackbar("Rotation", "Image", 0, 360, lambda x: None)

    while True:
        # get the rotation value from the trackbar
        rotation = cv.getTrackbarPos("Rotation", "Image")
        # get the rotated image
        rotated_img = rotate_image(img, rotation)
        # display the rotated image
        cv.imshow("Image", rotated_img)
        # break the loop when 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()