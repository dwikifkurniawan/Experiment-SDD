import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys

dir = "unbg ok"
# canny = "D:/1. KULIAH/MSIB/0. Perkuliahan Batch4/5. Image Processing/canny/"
d_dir = "Filtered/Sobel3 OK/"


# =================== Sobel Edge ================================

def main(argv):
    window_name = ('Sobel Demo - Simple Edge Detector')
    scale = 5
    delta = 3
    ddepth = cv.CV_16S

    # Load the image
    src = cv.imread(argv, cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image: ' + argv[0])
        return -1

    # Config_1
    src = cv.GaussianBlur(src, (5, 5), 0)
    src = cv.bilateralFilter(src, 9, 75, 75)

    # src = cv.GaussianBlur(src, (5, 5), 0)
    # src = cv.blur(src,(3,3))
    # src = cv.medianBlur(src,5)
    # src = cv.bilateralFilter(src,20,75,75)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # plt.subplot(121),plt.imshow(src,cmap = 'gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(grad,cmap = 'gray')
    # plt.title('Sobel Image'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # fourier(grad, sobe1)
    # sobel_1 = sobe1 + os.path.basename(argv)
    # cv.imwrite(sobel_1, grad)

    return grad


def fourier(img, dst, path):
    # img = cv.imread(raw, 0)
    assert img is not None, "file could not be read, check with os.path.exists()"
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # magnitude_spectrum = 50*np.log(np.abs(fshift))

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    fshift[crow - 15:crow + 16, ccol - 15:ccol + 16] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    ft = dst + os.path.basename(path)
    cv.imwrite(ft, img_back)


if __name__ == "__main__":
    for filename in os.scandir(dir):
        if filename.is_file():
            print(filename.path)
            # =================== Canny Edge ================================
            # img = cv.imread(filename.path,0)
            # edges = cv.Canny(img,50,50)
            # canny1 = canny + os.path.basename(filename.path)
            # cv.imwrite(canny1,edges)

            # plt.subplot(121),plt.imshow(img,cmap = 'gray')
            # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
            # plt.title('Canny Image'), plt.xticks([]), plt.yticks([])
            # plt.show()
            sobel = main(filename.path)
            fourier(sobel, d_dir, filename.path)
