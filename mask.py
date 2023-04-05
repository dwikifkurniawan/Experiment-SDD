# import cv2
# import numpy as np

# # Load the image
# img = cv2.imread('20230228_012939844_iOS.jpg')

# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply threshold
# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # Find contours
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Filter contours by area
# filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

# # Create mask
# mask = np.zeros_like(thresh)
# cv2.drawContours(mask, filtered_contours, -1, 255, cv2.FILLED)

# # Apply mask to original image
# result = cv2.bitwise_and(img, img, mask=mask)

# # Show the result
# result = cv2.resize(result, (756, 1008))
# cv2.imshow('Result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Load the image
img = cv2.imread('20230328_103438.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to the image to remove noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply a threshold to the image to binarize it
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Create a mask by finding contours in the thresholded image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(thresh)
cv2.drawContours(mask, contours, -1, 255, -1)

# Apply the mask to the original image
masked_img = cv2.bitwise_and(img, img, mask=mask)

# Display the result
img = cv2.resize(img, (640, 640))
mask = cv2.resize(mask, (640, 640))
masked_img = cv2.resize(masked_img, (640, 640))
cv2.imshow('Original Image', img)
cv2.imshow('Mask', mask)
cv2.imshow('Masked Image', masked_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
