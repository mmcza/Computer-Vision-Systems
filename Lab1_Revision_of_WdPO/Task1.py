import cv2 as cv
import os

def list_files(directory):
    files = os.listdir(directory)
    return files

def main():
    files_in_dir = list_files("./Pictures")
    print(files_in_dir)
    print("Press \"a\" to toggle on/off slideshow\nPress \"z\" to shorten time between pictures\nPress \"x\" to lengthen time between pictures")
    if files_in_dir:
        key = ord('s')
        next_pic_time = 20
        current_file = 0
        slide_show = False
        i = 0
        while key != ord('q'):
            # print(next_pic_time)
            if key == ord('a'):
                if slide_show:
                    slide_show = False
                else:
                    slide_show = True

            if key == ord('z') and next_pic_time > 1:
                next_pic_time -= 1

            if key == ord('x'):
                next_pic_time += 1

            current_picture = cv.imread("./Pictures/"+files_in_dir[current_file])
            cv.imshow("Picture", current_picture)
            if slide_show:
                i += 1
                if i >= next_pic_time:
                    i = 0
                    current_file += 1
                    if current_file >= len(files_in_dir):
                        current_file = 0
            key = cv.waitKey(10)

if __name__ == '__main__':
    main()
