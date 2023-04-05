import cv2


img = cv2.imread('NG/20230228_024824138_iOS.png')

# Apply bilateral filter with d = 15,
# sigmaColor = sigmaSpace = 75.
bilateral = cv2.bilateralFilter(img, 15, 75, 75)

# Save the output.
cv2.imwrite('Filtered/Bilateral/bilateral1.jpg', bilateral)