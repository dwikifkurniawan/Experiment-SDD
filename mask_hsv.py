import numpy as np
import cv2
import os
import fnmatch
from datetime import datetime

start = datetime.now()

dir = "D:/coba/circle/"
dest = "D:/coba/Filtered/CircleSmooth/"


def mask_hsv(pic, dst):
    # read image
    img = cv2.imread(pic)

    # Convert BGR to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

    # Image smoothing
    kernel = np.ones((7, 7), np.float32) / 15
    img_hsv = cv2.filter2D(img_hsv, -1, kernel)

    # define range of color in HSV
    lower_red = np.array([0, 0, 0])
    upper_red = np.array([255, 230, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)
    name = dst + os.path.basename(pic)
    cv2.imwrite(name, res)


count = len(fnmatch.filter(os.listdir(dir), '*.*'))
itr = 1
for filename in os.scandir(dir):
    if filename.is_file():
        print(filename.path)
        imgPath = filename.path
        mask_hsv(imgPath, dest)

        print(f"img masked: {itr} of {count}")
        itr += 1

# execution time
end = datetime.now()
td = (end - start).total_seconds() * 10 ** 3
print(f"The time of execution of the program is : {td:.03f}ms")


"""
def mask_hsv(pic, dst):
    img = cv2.imread(pic)
    # Convert BGR to HSV
    # img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

    kernel = np.ones((7, 7), np.float32) / 15
    img_hsv = cv2.filter2D(img_hsv, -1, kernel)

    # define range of color in HSV
    # lower_red = np.array([60, 128, 53])
    # lower_red = np.array([30, 50, 40])
    # upper_red = np.array([255, 255, 255])
    lower_red = np.array([0, 0, 0])
    upper_red = np.array([255, 75, 255])

    # # lower mask (0-10)
    # lower_red = np.array([0, 100, 20])
    # upper_red = np.array([10, 255, 255])
    # mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    #
    # # upper mask (170-180)
    # lower_red = np.array([160, 100, 20])
    # upper_red = np.array([179, 255, 255])
    # mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    #
    # # join my masks
    # mask = mask0 + mask1
    #
    # # set my output img to zero everywhere except my mask
    # output_img = img.copy()
    # output_img[np.where(mask == 0)] = 0
    #
    # # or your HSV image, which I *believe* is what you want
    # output_hsv = img_hsv.copy()
    # output_hsv[np.where(mask == 0)] = 0

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)
    name = dst + os.path.basename(pic)
    cv2.imwrite(name, res)

    # cv2.imwrite('Filtered/hsv/result3.png', res)
    # cv2.imwrite('Filtered/hsv/hls3.png', img_hsv)
    # cv2.imwrite('Filtered/hsv/mask3.png', mask)
    # print(img_hsv)
    # cv2.imshow("hsv", img_hsv)
    # cv2.waitKey()

"""