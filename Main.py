import cv2
import numpy as np

from Contour import find_contour
from OCR import extract_digit
from Solver.SolvingTable import SolvingTable
from Transform import transform

def test_solvingtable():
    table = [[-1 for i in range(9)] for j in range(9)]
    file = "D:\Dokumentumok\Visual Studio 2010\Projects\cs\Sudoku\sudoku5.csv"
    filestream = open(file)
    i = 0
    for line in filestream:
        strs = line[:-1].split(';')
        for j in range(9):
            if strs[j] == "":
                table[i][j] = -1
            else:
                table[i][j] = int(strs[j])

        i += 1
        if i >= 9:
            break
    st = SolvingTable(table)
    st.write()
    st.SolveBackTrack()
    st.write()

def get_knn():
    from glob import glob
    import os
    x = np.zeros((10, 224, 20, 20))

    for i in range(10):
        print i
        folder = "images/pdf2png/%d/" % i
        files = glob(folder + "*.png")
        for file in files:
            fname = os.path.basename(file)
            idx = int(fname[:4])
            img = cv2.imread(file, 0)
            x[i, idx, :, :] = img

    # Make it into a Numpy array. It size will be (50,100,20,20)
    # x = np.array(cells)

    # Now we prepare train_data and test_data.
    train = x[:, :112].reshape(-1, 400).astype(np.float32)  # Size = (1120,400)
    test = x[:, 112:224].reshape(-1, 400).astype(np.float32)  # Size = (1120,400)

    # Create labels for train and test data
    k = np.arange(10)
    train_labels = np.repeat(k, 112)[:, np.newaxis]
    test_labels = train_labels.copy()

    # Initiate kNN, train the data, then test it with test data for k=1
    knn = cv2.KNearest()
    knn.train(train, train_labels)
    return knn

if __name__ == '__main__':
    fname = """D:\dokumentumok\Python\PySudoku\images\img1_1_rot.png"""
    fname = """D:\dokumentumok\Python\PySudoku\images\img1_3.jpg"""
    # fname = """D:\dokumentumok\Python\PySudoku\images\extA.jpg"""
    fname = """D:/asztal/temp/sdk.jpg"""

    im = cv2.imread(fname)
    contour = find_contour(im)
    # cv2.drawContours(im, contour, -1, (255, 0, 0), 3)
    # cv2.imshow("", im)
    im2, trf = transform(im, contour)
    # cv2.imshow("trf", im2)

    knn = get_knn()
    step = 50
    table = [[0]*9 for i in range(9)]

    for i in range(9):
        for j in range(9):
            digit_img = im2[i * 50 : (i + 1) * 50, j * 50 : (j + 1) * 50]
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
            extr = extract_digit(digit_img, preserveAspectRatio=True, newsize=(20, 20))
            if extr is not None:
                # cv2.imshow("extr", extr)
                extr = 255 - extr.reshape((1, 400)).astype("float32")
                ret, result, neighbours, dist = knn.find_nearest(extr,k=6)
                print result, neighbours, dist
                table[i][j] = int(result[0, 0])
            else:
                print "empty"
                table[i][j] = '.'
            # cv2.imshow("digit", digit_img)
            # cv2.waitKey()

    for i in range(9):
        s = ""
        for j in range(9):
            s = s + str(table[i][j])
        print s

    for i in range(9):
        for j in range(9):
            if not (1 <= table[i][j] <= 9):
                table[i][j] = -1

    st = SolvingTable(table)
    st.SolveBackTrack()
    solved = st.getTable()
    for i in range(9):
        for j in range(9):
            if table[i][j] == -1:
                pos = np.array([[25 + 50 * j, 25 + 50 * i, 1]], dtype=float).T
                pos2 = np.linalg.inv(trf).dot(pos)
                pos2 /= pos2[2]
                text = str(solved[i][j])
                scale, thickness = 0.75, 2
                size, base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
                w, h = size
                pos2[0] -= w / 2.0
                pos2[1] += h / 2.0
                cv2.putText(im, text, (pos2[0], pos2[1]), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 128), thickness)

    cv2.imshow("slv", im)


    cv2.waitKey()
