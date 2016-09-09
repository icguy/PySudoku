import cv2
import numpy as np
from pprint import pprint

fname = """D:\dokumentumok\Python\PySudoku\images\img1_1_rot.png"""
fname = """D:\dokumentumok\Python\PySudoku\images\img1_6.jpg"""
fname = """D:\dokumentumok\Python\PySudoku\images\ext6.jpg"""

im = cv2.imread(fname)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
mblur = cv2.medianBlur(imgray, 5)
mmin, mmax = np.min(mblur), np.max(mblur)
mblur = np.uint8((mblur - mmin) * 255.0 / (mmax - mmin))

cv2.imshow("mbl", mblur)
cv2.waitKey()

thr = cv2.adaptiveThreshold(mblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 10)
# thr = cv2.Canny(mblur, 30, 150)

cv2.imshow("thr", thr)
cv2.waitKey()

dil = thr
kernel = np.ones((3, 3))
dil = cv2.dilate(thr, kernel)

cv2.imshow("dil", dil)
cv2.waitKey()

contours, hierarchy = cv2.findContours(dil,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print  len(contours)
contours = [c for c in contours if c.shape[0] > 3]
print  len(contours)
h, w, ch = im.shape
area = h * w
contours = [c for c in contours if cv2.contourArea(c) > area / 8]
print  len(contours)
contours = [cv2.approxPolyDP(c, 20, True) for c in contours]


cv2.drawContours(im, contours, -1, (255, 0, 0), 3)
cv2.imshow("im", im)
cv2.waitKey()