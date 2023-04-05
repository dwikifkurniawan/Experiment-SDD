import torch
import cv2
from PIL import Image
from ultralytics import YOLO

# rim.pt = model 2
# rim2.pt = model 1
model = YOLO("rim4.pt")
print(model)

# img_path = "D:/Soca AI/Research TMMIN/Converted/NG/20230228_015520430_iOS.png"
# results = model(img_path)
# print(results)

# im1 = Image.open(img_path)
results = model.predict(source="unbg", show=False, save=True)
# print(model.summary())

# img = cv2.resize(img, (640, 640))
#
# cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
# cv2.imshow("Detected Objects", combined_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()