from PIL import Image
import os
import cv2

dir = "D:/coba/Remove Background/static/results"
d_dir = "D:/coba/unbg ok"

for filename in os.scandir(dir):
    if filename.is_file():
        print(filename.path)
        # png_img = cv2.imread(filename.path)
        im = Image.open(filename.path)

        fill_color = (120, 8, 220)  # your new background color

        im = im.convert("RGBA")  # it had mode P after DL it from OP
        if im.mode in ('RGBA', 'LA'):
            background = Image.new(im.mode[:-1], im.size, fill_color)
            background.paste(im, im.split()[-1])  # omit transparency
            im = background

        # bg = Image.new("RGB", img.size, (255, 255, 255))
        # bg.paste(img, img)
        # img = img.convert('RGB')

        fname = os.path.basename(filename.path)
        fname2 = os.path.splitext(fname)
        # print(fname2[0])

        # cv2.imwrite(d_dir + fname2[0] + '.jpg', png_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        im.convert("RGB").save(d_dir + "/" + fname2[0] + ".jpg")