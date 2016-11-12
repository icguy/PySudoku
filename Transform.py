import cv2
import numpy as np

def order_pts(contour):
    """
    :param img:
    :param contour:
    :return: contour points in order: top left, top right, bottom left, bottom right
    """
    pts = contour[0].reshape((4, 2))
    cog = np.sum(pts, 0) / 4
    ideals = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    pts2 = pts - cog * np.ones((4, 1), int)

    corresp = pts2.dot(ideals.T)
    # print corresp
    indices = np.argmax(corresp, 0)
    return np.array([pts[indices[i],:] for i in range(4)], dtype="float32").reshape((4, 2))

def transform(img, contour, newsize = (450, 450)):
    src_pts = order_pts(contour)

    w, h = newsize
    dest_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype="float32")
    print src_pts
    print dest_pts
    trf = cv2.getPerspectiveTransform(src_pts, dest_pts)
    return cv2.warpPerspective(img, trf, newsize), trf

if __name__ == '__main__':
    transform(None, [np.array([[-5, -5], [4, -6], [-3, 5], [4, 4]])])