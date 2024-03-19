import cv2 as cv

def main():
    # link to image https://cdn2.vectorstock.com/i/1000x1000/16/91/cat-and-dog-seamless-pattern-cute-cartoon-style-vector-37171691.jpg
    img = cv.imread('cat-and-dog-pattern.jpg')
    img = cv.resize(img, (0, 0), fx=0.75, fy=0.75)
    cv.imshow('Image', img)
    # create a template from the image
    template = img.copy()
    template = template[60:240, 20:170]
    cv.imshow('Template', template)
    # match the template with the image
    result = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
    ret, result_threshold = cv.threshold(result, 0.3, 1., cv.THRESH_BINARY)
    cv.imshow('Result', result_threshold)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
