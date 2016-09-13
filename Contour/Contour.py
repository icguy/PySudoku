import cv2
import numpy as np
from pprint import pprint
import random

DISP = True
WAIT = False

def disp(name, img, wait = True):
    if DISP:
        cv2.imshow(name, img)
    if wait and WAIT:
        cv2.waitKey()

if __name__ == '__main__':

    fname = """D:\dokumentumok\Python\PySudoku\images\img1_1_rot.png"""
    fname = """D:\dokumentumok\Python\PySudoku\images\img1_6.jpg"""
    fname = """D:\dokumentumok\Python\PySudoku\images\ext4.jpg"""

    im = cv2.imread(fname)
    minsize = min(*im.shape[:2])
    print "minsize: %d" % minsize

    kernel_size = 5 if minsize > 600 else 3

    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    mblur = cv2.medianBlur(imgray, kernel_size)
    mmin, mmax = np.min(mblur), np.max(mblur)
    mblur = np.uint8((mblur - mmin) * 255.0 / (mmax - mmin))

    disp("mbl", mblur)

    thr = cv2.adaptiveThreshold(mblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, kernel_size, 10)
    thr2 = cv2.Canny(mblur, 30, 150)
    thr = thr | thr2

    disp("thr", thr)

    dil = thr
    kernel = np.ones((3, 3))
    dil = cv2.dilate(thr, kernel)

    disp("dil", dil)

    contours, hierarchy = cv2.findContours(dil,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print  len(contours)
    contours = [c for c in contours if c.shape[0] > 3]
    print  len(contours)
    h, w, ch = im.shape
    area = h * w
    contours = [c for c in contours if cv2.contourArea(c) > area / 8]
    print  len(contours)
    contours = [cv2.approxPolyDP(c, 20, True) for c in contours]
    print contours

    marker_corners = np.array([[0,0],[1, 0], [1,1], [0,1]], dtype=np.float32).reshape(-1, 1, 2)
    for c in contours:
        H, mask = cv2.findHomography(np.float32(c).reshape(-1, 1, 2), marker_corners)
        num_pt = marker_corners.shape[0]
        corners = np.ones((num_pt, 3), np.float32)
        marker_corners2 = np.copy(marker_corners).reshape(num_pt, 2)
        corners[:, :2] = marker_corners2
        print H, corners
        new_c = corners.dot(np.linalg.inv(H.T))
        for i in range(new_c.shape[0]):
            new_c[i,:] /=new_c[i,2]
        print new_c

    cv2.drawContours(im, contours, -1, (255, 0, 0), 3)
    disp("im", im)