# Import packages
import cv2
import numpy as np

img = cv2.imread('D:/1. KULIAH/MSIB/0. Perkuliahan Batch4/5. Image Processing/CircleSmooth/20230228_012939844_iOS.png')
print(img.shape)  # Print image shape

x = 0
y = 0
slice = 256
while y < img.shape[0] - slice:
    while x < img.shape[1] - slice:
        cropped_image = img[y:y + slice, x:x + slice]

        cv2.imshow("cropped", cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # print("x = ", x)
        # print("y = ", y)

        x = int(x + (slice * (1 - 0.2)))

        if x + slice > img.shape[1]:
            x = img.shape[1] - slice

    x = 0
    y = int(y + (slice * (1 - 0.2)))
    if y + slice > img.shape[0]:
        y = img.shape[0] - slice
