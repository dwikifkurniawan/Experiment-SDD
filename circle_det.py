import cv2
import numpy as np
import os

def circle_det(raw, dst):
    images = cv2.imread(raw, cv2.IMREAD_COLOR)
    white_color = (255, 255, 255)
    mask = np.zeros_like(images)

    radius = 1400

    gray_images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray_images, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 80, param1=90,
                                        param2=50, minRadius=radius - 10,
                                        maxRadius=radius + 10)  # 80, param1 = 90, param2 = 30, minRadius = 1350, maxRadius = 1370
    print(detected_circles)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        print(detected_circles[0])
        # detected_circles[0] = [1374, 1746, 1341]
        # detect how many circles detected
        # temp=detected_circles[0]
        # print (temp[0])

        # manual / 1484 1696 1387
        # a, b, r = 1490, 1690, 1410
        a, b, r = 1500, 1464, 1450
        mask = cv2.circle(mask, (a, b), r, white_color, -1)

        print(len(detected_circles[0]))
        # for pt in detected_circles[0, :]:
        #     a, b, r = pt[0], pt[1], pt[2]
        #     # cv2.circle(images, (a,b), r, (0, 255, 0), 2) #Drawing circle
        #     mask = cv2.circle(mask, (a, b), r, white_color, -1)

    result = cv2.bitwise_and(images, mask)
    # gray_res = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    goals = dst + os.path.basename(raw)
    cv2.imwrite(goals, result)

raw = "20230328_103438.jpg"
dst = "circle_"
circle_det(raw, dst)