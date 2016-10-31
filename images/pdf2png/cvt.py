import cv2
import numpy as np
from OCR import *
from pprint import pprint
from glob import glob
import os

for i in range(10):
    folder = "%d/" % i
    files = glob(folder + "*.png")
    for file in files:
        fname = os.path.basename(file)
        fname2 = fname.rjust(8, "0")
        img = cv2.imread(file, 0)
        img2 = extract_digit(img)
        cv2.imshow("", img2)
        cv2.waitKey()
