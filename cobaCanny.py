import os
import cv2
import numpy as np

# dir = "NG/"
dir = "Filtered/Sobel2"
filter_dir = "Filtered/Sobel + Canny/"


def filtering(argv, output):
    img = cv2.imread(argv, 0)

    # # Setting parameter values
    lower_thresh = 0  # Lower Threshold
    upper_thresh = 50  # Upper threshold

    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # v = np.median(img)
    # sigma = 0.33
    #
    # # ---- apply optimal Canny edge detection using the computed median----
    # lower_thresh = int(max(0, (1.0 - sigma) * v))
    # upper_thresh = int(min(255, (1.0 + sigma) * v))

    # Applying the Canny Edge filter
    edge = cv2.Canny(img, lower_thresh, upper_thresh)

    name = output + os.path.basename(argv)
    cv2.imwrite(name, edge)

    return 0


# img = cv2.imread('NG')

if __name__ == "__main__":
    for filename in os.scandir(dir):
        if filename.is_file():
            print(filename.path)

            filtering(filename.path, filter_dir)

