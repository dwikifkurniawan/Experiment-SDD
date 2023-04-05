from PIL import Image
import os
import cv2
import numpy as np

dir = "D:/coba/unbg ok"
d_dir = "D:/coba/Filtered/InpaintTelea/"
d_dir2 = "D:/coba/Filtered/InpaintNS/"

for filename in os.scandir(dir):
    if filename.is_file():
        # read image
        print(filename.path)
        img = cv2.imread(filename.path)
        hh, ww = img.shape[:2]
        print("read image")

        # threshold
        lower = (150, 150, 150)
        upper = (240, 240, 240)
        thresh = cv2.inRange(img, lower, upper)
        print("threshold")

        # apply morphology close and open to make mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel, iterations=1)
        print("morphology")

        # floodfill the outside with black
        black = np.zeros([hh + 2, ww + 2], np.uint8)
        mask = morph.copy()
        mask = cv2.floodFill(mask, black, (0, 0), 0, 0, 0, flags=8)[1]
        print("floodfill")

        # use mask with input to do inpainting
        result1 = cv2.inpaint(img, mask, 101, cv2.INPAINT_TELEA)
        result2 = cv2.inpaint(img, mask, 101, cv2.INPAINT_NS)
        print("inpainting")


        # saturated = Binarize[ColorConvert[img, "Grayscale"], .9]

        name1 = d_dir + os.path.basename(filename.path)
        name2 = d_dir2 + os.path.basename(filename.path)
        cv2.imwrite(name1, result1)
        cv2.imwrite(name2, result2)
        print("write")
        # print(fname2[0])
