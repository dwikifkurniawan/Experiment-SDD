import os
import cv2

dir = "NG/"
filter_dir = "Filtered/CLAHE2/"


def filtering(argv, output):
    img = cv2.imread(argv, 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(img)

    name = output + os.path.basename(argv)
    cv2.imwrite(name, cl)

    return 0


# img = cv2.imread('NG')

if __name__ == "__main__":
    for filename in os.scandir(dir):
        if filename.is_file():
            print(filename.path)

            filtering(filename.path, filter_dir)
