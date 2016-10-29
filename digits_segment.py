import cv2
import numpy as np
from OCR import *
from pprint import pprint

out_folder = "D:/Asztal/temp/0123456789 - Copy/pdf2png/"
img = cv2.imread("D:/Asztal/temp/0123456789 - Copy/pdf2png/0123456789.png", 0)
img = 255 - img
rows = np.max(img, 1)
rows[rows != 0] = 255
row_indices = []
for i in range(1, rows.size):
    if rows[i - 1] != rows[i]:
        row_indices.append(i)

nextidx = [0] * 10

for i in range(len(row_indices)):
    if i % 2 == 0:
        continue
    idx1 = row_indices[i - 1]
    idx2 = row_indices[i]

    part = img[(idx1 - 2) : (idx2 + 2), :]
    part_orig = part.copy()
    part[part != 0] = 255
    indices, val = indexObjects(part)
    print i, len(row_indices), val

    if val % 10 != 1:
        cv2.imshow("", part)
        cv2.waitKey()
    else:
        bboxes = []
        for j in range(1, val):
            bbox = get_bounding_box(indices, j)
            bboxes.append(bbox)

        bboxes.sort(key=lambda bb:bb[2])

        for j in range(len(bboxes)):
            uu, dd, ll, rr = bboxes[j]
            digit_img = part_orig[uu: dd, ll: rr]
            digit_num = j % 10
            folder = out_folder + str(digit_num)
            filename = folder + "/" + str(nextidx[digit_num]) + ".png"
            cv2.imwrite(filename, digit_img)
            nextidx[digit_num] = nextidx[digit_num] + 1


            # cv2.imshow(str((j % 10)), part_orig[uu: dd, ll: rr])
            # cv2.waitKey()
            # cv2.destroyAllWindows()

