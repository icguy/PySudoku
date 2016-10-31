import cv2
import numpy as np
from pprint import pprint

DISP = False
WAIT = False

def disp(name, img, wait = True):
    if DISP:
        cv2.imshow(name, img)
    if wait and WAIT:
        cv2.waitKey()

def normalize(img):
    mmin, mmax = np.min(img), np.max(img)
    return np.uint8((img - mmin) * 255.0 / (mmax - mmin))

def find_contour(im):
    maxsize = max(*im.shape[:2])
    scale = 800.0 / maxsize
    im = cv2.resize(im, None, fx = scale, fy = scale)
    print im.shape

    kernel_size = 5

    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    imgray = normalize(imgray)
    mblur = imgray
    mblur = cv2.medianBlur(imgray, kernel_size)
    mblur = normalize(mblur)

    disp("mbl", mblur)

    thr = cv2.adaptiveThreshold(mblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, kernel_size, 10)
    kernel = np.ones((3, 3))
    thr = cv2.erode(thr, kernel)
    thr2 = cv2.Canny(mblur, 30, 150)
    thr = thr | thr2

    disp("thr", thr)

    dil = thr
    kernel = np.ones((3, 3))
    # dil = cv2.dilate(thr, kernel)
    # dil = cv2.erode(dil, kernel)

    disp("dil", dil)

    contours = extract_contours(dil)
    if len(contours) == 0:
        dil = cv2.dilate(dil, kernel)
        contours = extract_contours(dil)



    # marker_corners = np.array([[0,0,0],[1, 0, 0], [1,1, 0], [0,1, 0]], dtype=np.float32)
    # for c in contours:
    #     c = np.float32(c.reshape(-1, 2))
    #     rv, rvec, tvec = cv2.solvePnP(marker_corners, c, np.eye(3), None, flags=cv2.CV_ITERATIVE)
    #     print tvec, rvec

    newcontours = map(lambda c : (c / scale).astype("int"), contours)
    if len(newcontours) == 0:
        return []

    return [max(newcontours, key = lambda c : cv2.contourArea(c))]


def extract_contours(dil):
    contours, hierarchy = cv2.findContours(dil, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print  len(contours)
    # contours = [c for c in contours if c.shape[0] > 3]
    print  len(contours)
    h, w = dil.shape
    area = h * w
    contours = [c for c in contours if cv2.contourArea(c) > area / 8]
    print  len(contours)
    contours = [cv2.approxPolyDP(c, 20, True) for c in contours]
    print  len(contours)
    contours = [c for c in contours if c.shape[0] == 4]
    print  len(contours)
    return contours


if __name__ == '__main__':

    fname = """D:\dokumentumok\Python\PySudoku\images\img1_1_rot.png"""
    fname = """D:\dokumentumok\Python\PySudoku\images\img1_6.jpg"""
    fname = """D:\dokumentumok\Python\PySudoku\images\ext1.jpg"""

    im = cv2.imread(fname)
    contours = find_contour(im)
    print contours

    cv2.drawContours(im, contours, -1, (255, 0, 0), 3)
    disp("im", im)
    if not WAIT:
        cv2.waitKey()