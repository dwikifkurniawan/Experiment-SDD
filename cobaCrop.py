# Import packages
import cv2
import numpy as np
from ultralytics import YOLO

# img = cv2.imread('FpLShLQacAIZ1hp.jpg')
# img = cv2.imread('c.png')
img = cv2.imread('20230228_020757465_iOS.png')
print(img.shape)  # Print image shape

model = YOLO("best (1).pt")
# results = model.predict(source="unbg", show=False, save=True)

x = 0
y = 0
slice = 256
x_max = img.shape[1]
y_max = img.shape[0]
while y < y_max:
    while x < x_max:
        print("X2 = ", x)
        if x > x_max:
            # print("masik if")
            # x = img.shape[0] - slice
            cropped_image = img[y:y + slice, x_max - slice:x_max]
            # break
        else:
            # print("masuk elif")
            cropped_image = img[y:y + slice, x:x + slice]
            x = x + slice

        # cv2.imshow("cropped", cropped_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        res = model(cropped_image)
        # boxes =
        print(res[0].boxes)
        res_plotted = res[0].plot()
        # cv2.imshow("result", res_plotted)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        print("x = ", x)
        print("y = ", y)
        # print("shape ", img.shape[0])

    x = 0
    if y + slice > y_max:
        y = y_max - slice
    else:
        y = y + slice
