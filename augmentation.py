import os
import cv2
import glob
from scipy import ndimage

names = ["takahashi", "tamaki"]
os.makedirs("./test", exist_ok=True)

for name in names:
    in_dir = "./face/" + name + "/*"
    out_dir = "./train/" + name
    os.makedirs(out_dir, exist_ok=True)
    in_jpg = glob.glob(in_dir)

    for i in range(len(in_jpg)):
        img = cv2.imread(str(in_jpg[i]))
        for ang in range(-20, 20, 5):
            # Rotation
            img_rot = ndimage.rotate(img, ang)
            img_rot = cv2.resize(img_rot, (64, 64))
            filename = os.path.join(out_dir, str(i) + "_" + str(ang) + ".jpg")
            cv2.imwrite(str(filename), img_rot)
            # Thresholding
            img_thr = cv2.threshold(img_rot, 100, 255, cv2.THRESH_TOZERO)[1]
            filename = os.path.join(out_dir, str(i) + "_" + str(ang) + "thr.jpg")
            cv2.imwrite(str(filename), img_thr)
            # Gaussian filter
            img_filter = cv2.GaussianBlur(img_rot, (5,5), 0)
            filename = os.path.join(out_dir, str(i) + "_" + str(ang) + "filter.jpg")
            cv2.imwrite(str(filename), img_filter)
