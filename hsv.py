import cv2
import numpy as np
import os 

def replace(arr1,arr2):
    # total = 0
    itr_1=0
    for x in arr2:
        itr_2=0
        for y in x:
            if (y>0).any() :
                arr1[itr_1][itr_2] = 0
                # total = total+1
            itr_2=itr_2+1
        itr_1=itr_1+1
    # print (total)
    return arr1

def match(raw,dst):
    img_rgb = cv2.imread(raw)
    assert img_rgb is not None, "file could not be read, check with os.path.exists()"

    white_color = (255, 255, 255)
    mask = np.zeros_like(img_rgb)

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(dst+'match.png', cv2.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        x, y = pt[0], pt[1]
        cv2.rectangle(img_rgb, pt, (x + w, y + h), (0,255,0), 2)
        mask = cv2.rectangle(mask, pt, (x + w, y + h), white_color, -1)
    
    result = cv2.bitwise_and(img_rgb, mask)

    goals = dst + 'match_' + os.path.basename(raw)
    cv2.imwrite(goals,img_rgb)
    # sobel(result,raw,dst)

def fourier(img,img_mask,raw,dst):
    # img = cv2.imread(raw, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow,ccol = rows//2 , cols//2
    fshift[crow-15:crow+16, ccol-15:ccol+16] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    # result = cv2.addWeighted(img_back,0.5,img_mask,1,0)
    # result = cv2.bitwise_and(img_back,img_back, mask=img_mask)
    result = replace(img_back,img_mask)

    goals = dst + '3_3_' + os.path.basename(raw)
    cv2.imwrite(goals,result)
    # return goals,img_back
    # fourier_2(result,raw,dst)

def sobel(img_mask, raw, dst):
    scale = 7
    delta = 3
    ddepth = cv2.CV_16S
    
    # Load the image
    src = cv2.imread(raw, cv2.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + raw[0])
        return -1
    
    # Config_1
    src = cv2.GaussianBlur(src, (5, 5), 0)
    src = cv2.bilateralFilter(src,9,75,75)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Gradient-X 
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    # print("grad_type : ", type(grad))
    # print("img_mask_type : ", type(img_mask))
    # result = cv2.addWeighted(grad,0.5,img_mask,1,0)
    # result = replace(grad,img_mask)
        
    goals = dst + '2_2_' + os.path.basename(raw)
    # cv2.imwrite(goals,grad)
    fourier(grad,img_mask,raw,dst)
    # return grad,img_mask,raw,dst

def mask_light(raw,dst):
    img = cv2.imread(raw,cv2.IMREAD_GRAYSCALE)
    # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=img+(np.ones((img.shape))+25)
    img[img>255]=255

    # define range of color
    lower = np.array([245])
    upper = np.array([255])

    # Threshold the  image to get only colors is defined
    mask = cv2.inRange(img, lower, upper)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask=mask)

    # result = res[res > 170] = 160
    # result = result[result < 150] = 0
    # result = res*0.2
    # result =  np.uint16(np.around(result))

    # kernel = np.ones((15,15),np.float32)/15
    # res = cv2.filter2D(res,-1,kernel)

    goals = dst + '1_1_' + os.path.basename(raw)
    # cv2.imwrite(goals,res)
    sobel(res,raw,dst)
    # return result,raw,dst

def circle_det(raw,dst):
    images = cv2.imread(raw, cv2.IMREAD_COLOR)
    white_color = (255, 255, 255)
    mask = np.zeros_like(images)

    radius = 1340
    
    gray_images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray_images, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred, 
                   cv2.HOUGH_GRADIENT, 1, 80, param1 = 90,
               param2 = 30, minRadius = radius-10, maxRadius = radius+10) #80, param1 = 90, param2 = 30, minRadius = 1350, maxRadius = 1370
    print (detected_circles)
    
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        print (detected_circles[0])
        # detected_circles[0] = [1374, 1746, 1341]
        #detect how many circles detected
        # temp=detected_circles[0]
        # print (temp[0])

        # manual / 1484 1696 1387
        # a, b, r = 1490, 1690, 1410
        # mask = cv2.circle(mask, (a, b), r, white_color, -1)

        print(len(detected_circles[0]))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            # cv2.circle(images, (a,b), r, (0, 255, 0), 2) #Drawing circle
            mask = cv2.circle(mask, (a, b), r, white_color, -1)

    result = cv2.bitwise_and(images, mask)
    # gray_res = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    goals = dst + os.path.basename(raw)
    cv2.imwrite(goals,result)
    # match(goals,dst)
    # mask_light(goals,dst)
    # return goals, result

path_img = 'D:/1. KULIAH/MSIB/0. Perkuliahan Batch4/Dataset/04. White/20230228_024646073_iOS.png' #20230228_012939844_iOS.png
path_dst = 'D:/1. KULIAH/MSIB/0. Perkuliahan Batch4/5. Image Processing/hsv/'
path_circle = 'D:/1. KULIAH/MSIB/0. Perkuliahan Batch4/5. Image Processing/circle_1/20230228_024646073_iOS.png'

# circle_det(path_img, path_circle)

# goals,dst = circle_det(path_img, path_dst)
# result,raw,dst = mask_light(goals,dst)
# print(result)
# grad,img_mask,raw,dst = sobel(result,raw,dst)
# print(grad)
# goals,img_back = fourier(grad,img_mask,raw,dst)
# print(img_back)

# for filename in os.scandir(path_circle):
#         if filename.is_file():
#             print(filename.path)
#             match(filename.path, path_dst)

match(path_circle, path_dst)