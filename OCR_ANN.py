import numpy as np
import cv2


def inBounds(img, r, c):
    h, w = img.shape[0], img.shape[1]
    return 0 <= c < w and 0 <= r < h

def indexObjects(img):
    indices = np.zeros(img.shape, 'uint8')
    toCheck = []

    h, w = img.shape
    newIdx = 1

    for r in range(h):
        for c in range(w):
            if img[r, c] > 0 and indices[r, c] == 0:
                toCheck.append((c, r))
                while len(toCheck) > 0:
                    cc, cr = toCheck[0]
                    if not inBounds(img, cr, cc):
                        toCheck.pop(0)
                        continue

                    if img[cr, cc] > 0 and indices[cr, cc] == 0:
                        indices[cr, cc] = newIdx
                        toCheck.append((cc - 1, cr))
                        toCheck.append((cc + 1, cr))
                        toCheck.append((cc, cr - 1))
                        toCheck.append((cc, cr + 1))

                    toCheck.pop(0)

                newIdx += 1
    return indices, newIdx

def test():
    img = cv2.imread("""D:/dokumentumok/opencv/sources/samples/python2/data/digits.png""", 0)
    img2 = img
    for i in range(50):
        for j in range(100):
            pic = img[i * 20 : (i+1) * 20, j * 20 : (j+1) * 20]
            pic = 255 - extract_digit(pic, True, True)
            # cv2.imshow("", pic)
            # cv2.waitKey()
            img2[i * 20 : (i + 1) * 20, j * 20 : (j + 1) * 20] = pic
    cv2.imwrite("digits4.png", img2)


    img = cv2.imread("""images/9_1.png""", 0)
    img = cv2.imread("""images/1.png""", 0)
    # img = cv2.imread("""images/img1_2.jpg""", 0)
    # img = cv2.imread("""images/blob_test.png""", 0)
    # img = cv2.imread("""images/blob_test2.png""", 0)
    # img = cv2.imread("""images/blob_test3.png""", 0)

    img = extract_digit(img)

    cv2.imshow("fn.png", img)
    cv2.waitKey()

def extract_digit(img, invert = False, skip_bg = False):
    """
    :param img:
    :param invert: True if background is black, and foreground/digit is white
    :param skip_bg: does not compute background removal, should set True if only digit is present, no other lines
    :return:
    """
    h, w = img.shape
    if invert: img = 255 - img
    mmin, mmax = np.min(img), np.max(img)
    img_norm = np.uint8((img - mmin) * 255.0 / (mmax - mmin))
    r, img = cv2.threshold(img_norm, 128, 255, cv2.THRESH_BINARY)

    if not skip_bg:
        # finding largest background
        indices, num = indexObjects(img)
        sizes = [(indices == i).sum() for i in range(1, num)]
        maxSize = max(sizes)
        maxIdx = sizes.index(maxSize) + 1

        largeBg = np.zeros_like(img)
        largeBg[indices == maxIdx] = 255
        outside = np.zeros((h + 2, w + 2), "uint8")

        # filling outside
        for row in range(h):
            if outside[row, 0] == 0 and largeBg[row, 0] == 0:
                cv2.floodFill(largeBg, outside, (0, row), 255, 0, 0, cv2.FLOODFILL_MASK_ONLY)
                # print  outside
            if outside[row, w - 1] == 0 and largeBg[row, w - 1] == 0:
                cv2.floodFill(largeBg, outside, (w - 1, row), 255, 0, 0, cv2.FLOODFILL_MASK_ONLY)
                # print  outside
        for col in range(w):
            if outside[0, col] == 0 and largeBg[0, col] == 0:
                cv2.floodFill(largeBg, outside, (col, 0), 255, 0, 0, cv2.FLOODFILL_MASK_ONLY)
                # print  outside
            if outside[h - 1, col] == 0 and largeBg[h - 1, col] == 0:
                cv2.floodFill(largeBg, outside, (col, h - 1), 255, 0, 0, cv2.FLOODFILL_MASK_ONLY)
                # print  outside

        mmin, mmax = np.min(outside), np.max(outside)
        if mmax > 0:
            outside = np.uint8((outside - mmin) * 255.0 / (mmax - mmin))
        outside = outside[1:-1, 1:-1]

        # erasing bg
        background = largeBg + outside
        img = img + background
    img = 255 - img

    # cv2.imshow("i", img)
    # cv2.waitKey()


    # bounding box
    U, D, L, R = get_bounding_box(img)

    img = warp_bounding(U, D, L, R, img_norm)
    return img

def get_bounding_box(img):

    h, w = img.shape
    U = h
    D = 0
    L = w
    R = 0
    for r in range(h):
        for c in range(w):
            if img[r, c] == 255:
                if r < U:
                    U = r
                if r > D:
                    D = r
                if c < L:
                    L = c
                if c > R:
                    R = c

    return U, D, L, R

def warp_bounding(U, D, L, R, img_norm, newsize = (20, 20)):
    w, h = newsize
    side = max(D-U, R-L)
    centerx, centery = ((L + R)/2.0, (U + D)/2.0)
    U = centery - side / 2
    D = centery + side / 2
    L = centerx - side / 2
    R = centerx + side / 2
    # print U, D, L, R

    pts1 = np.float32([[L, U], [R, U], [R, D]])
    pts2 = np.float32([[0, 0], [w, 0], [w, h]])
    trf = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img_norm, trf, (w, h))

def get_cog(img):
    h, w = img.shape

    cx, cy = 0.0, 0.0
    num = 0
    for r in range(h):
        for c in range(w):
            if img[r, c] == 255:
                num += 1
                cx += c
                cy += r
    cx /= num
    cy /= num
    dx = w / 2.0 - cx
    dy = h / 2.0 - cy
    return dx, dy

if __name__ == '__main__':
    # test()
    # exit()

    img = cv2.imread("""D:\dokumentumok\Python\PySudoku\digits4.png""")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

    # Make it into a Numpy array. It size will be (50,100,20,20)
    x = np.array(cells)

    # Now we prepare train_data and test_data.
    train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
    test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

    # Create labels for train and test data
    k = np.arange(10)
    train_labels = np.repeat(k,250)[:,np.newaxis]
    test_labels = train_labels.copy()

    ann_labels = np.zeros((train_labels.size, 10), dtype="float64")
    for i in range(train_labels.size):
        ann_labels[i, train_labels[i, 0]] = 1
    ann = cv2.ANN_MLP(np.array([400, 80, 10]))
    ann.train(train, ann_labels, None)

    # Initiate kNN, train the data, then test it with test data for k=1
    knn = cv2.KNearest()
    knn.train(train,train_labels)
    print "train shape", train.shape
    print "test shape", test.shape
    ret,result,neighbours,dist = knn.find_nearest(test,k=5)

    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print accuracy
    print correct, result.size

    for i in range(1, 10):
        img = cv2.imread("""images/%d.png""" % i, 0)
        img = extract_digit(img)
        img = 255-img
        cv2.imshow("i", img)
        cv2.waitKey()
        img = np.float32(img.reshape((1, 400)))
        ret,result,neighbours,dist = knn.find_nearest(img,k=10)

        print ann.predict(img)
        print result