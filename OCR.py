import cv2
import numpy as np

DISP = False
def disp(title, img):
    if DISP:
        cv2.imshow(title, img)

def inBounds(img, r, c):
    h, w = img.shape[0], img.shape[1]
    return 0 <= c < w and 0 <= r < h

def indexObjects(img):
    """
    :param img:
    :return: (indices, newidx), where indices is the index array.
    0 is the background and the objects are labeled with indices going from 1 to newidx-1, inclusive
    """
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
                        toCheck.append((cc - 1, cr - 1))
                        toCheck.append((cc + 1, cr + 1))
                        toCheck.append((cc + 1, cr - 1))
                        toCheck.append((cc - 1, cr + 1))

                    toCheck.pop(0)

                newIdx += 1
    return indices, newIdx

def test():
    img = cv2.imread("""images/9_1.png""", 0)
    img = cv2.imread("""images/9_2.png""", 0)
    # img = cv2.imread("""images/img1_2.jpg""", 0)
    # img = cv2.imread("""images/blob_test.png""", 0)
    # img = cv2.imread("""images/blob_test2.png""", 0)
    # img = cv2.imread("""images/blob_test3.png""", 0)

    img = extract_digit(img)

    cv2.imshow("fn.png", img)
    cv2.waitKey()

def get_gaussian(size):
    h, w = size
    kernel_horiz = cv2.getGaussianKernel(w if w % 2 == 1 else w + 1, -1)
    kernel_vert = cv2.getGaussianKernel(h if h % 2 == 1 else h + 1, -1)
    kernel = kernel_vert * kernel_horiz.T

    return kernel[0:h, 0:w]

def extract_digit(img, invert = False, skip_bg = False, preserveAspectRatio = False, newsize=(20, 20)):
    """
    :param img: should be grayscale
    :param invert: True if background is black (zero), foreground is white (255)
    :param skip_bg: set True if there's only the digit itself in the image no other BLOB-s
    :return: the extracted digit image or None if digit not present
    """
    h, w = img.shape
    if invert: img = 255 - img
    mmin, mmax = np.min(img), np.max(img)
    img_norm = np.uint8((img - mmin) * 255.0 / (mmax - mmin))
    r, img = cv2.threshold(img_norm, 128, 255, cv2.THRESH_BINARY)
    disp("thr", img)

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
        img[background == 255] = 255
        disp("bbox", img)

        #searching for biggest blob
        positive_img = 255 - img
        disp("positive_img", positive_img)
        blob_indices, num = indexObjects(positive_img)
        if num < 2:
            return None # no blobs found

        # disp("blob indices", normalize(blob_indices))
        dil = cv2.dilate(positive_img, (3, 3))
        # dil = positive_img
        img_weighted = np.multiply(dil, get_gaussian(dil.shape))
        sizes = [img_weighted[blob_indices == i].sum() for i in range(1, num)]
        maxSize = max(sizes)
        maxIdx = sizes.index(maxSize) + 1
        if maxSize < 2:
            return None # nothing big in the middle

        img[blob_indices != maxIdx] = 255
        disp("final img", img)

    img = 255 - img

    # bounding box
    udlr = get_bounding_box(img)
    if udlr is None:
        return None
    U, D, L, R = udlr
    print D-U, R-L
    if min(D-U, R-L) < 4:
        return None # too little bounding box, probably false detection

    img = warp_bounding(U, D, L, R, img_norm,newsize, preserveAspectRatio, 255 if invert else 0)
    return img

def get_bounding_box(img, value = 255):
    """

    :param img:
    :param value: the value to search for
    :return: U, D, L, R params of bounding rectangle, nonne, if image is blank
    """
    h, w = img.shape
    U = h
    D = 0
    L = w
    R = 0
    found = False
    for r in range(h):
        for c in range(w):
            if img[r, c] == value:
                found = True
                if r < U:
                    U = r
                if r > D:
                    D = r
                if c < L:
                    L = c
                if c > R:
                    R = c
    if not found: return None
    return U, D, L, R

def warp_bounding(U, D, L, R, img_norm, newsize = (20, 20), preserveAspectRatio = False, bgColor = 0):
    w, h = newsize
    if preserveAspectRatio:
        side = max(D-U, R-L)
        centerx, centery = ((L + R)/2.0, (U + D)/2.0)
        U = centery - side / 2
        D = centery + side / 2
        L = centerx - side / 2
        R = centerx + side / 2

    pts1 = np.float32([[L, U], [R, U], [R, D]])
    pts2 = np.float32([[0, 0], [w, 0], [w, h]])
    trf = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img_norm, trf, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bgColor)

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

    img = cv2.imread("""D:\dokumentumok\Python\PySudoku\digits3.png""")
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

    # Initiate kNN, train the data, then test it with test data for k=1
    knn = cv2.KNearest()
    knn.train(train,train_labels)
    print test.shape
    ret,result,neighbours,dist = knn.find_nearest(test,k=5)

    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print accuracy
    print correct, result.size


    img = cv2.imread("""images/9_3.png""", 0)
    img = extract_digit(img)
    img = 255-img
    cv2.imshow("i", img)
    cv2.waitKey()
    img = np.float32(img.reshape((1, 400)))
    ret,result,neighbours,dist = knn.find_nearest(img,k=5)
    print result