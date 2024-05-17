import os
import cv2 as cv
from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser("Converter")

parser.add_argument("--source_path", "-s", required=True,type=str)



args= parser.parse_args()
input_dir = args.source_path + "/input/"

images = os.listdir(input_dir)
kps = []
descs = []
image_count = 0
for image in images:
    file = os.path.join(input_dir, image)
    img = cv.imread(file)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp, desc = sift.detectAndCompute(img, None)
    kps.append(kp)
    descs.append(desc)

    img=cv.drawKeypoints(gray_img,kp,img)
 
    cv.imshow("sift points", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    image_count += 1


matcher = cv.BFMatcher(crossCheck=True)
matches = matcher.match(descs[0], descs[1])




