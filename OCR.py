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
    img = cv2.imread("""images/9_1.png""", 0)
    mmin, mmax = np.min(img), np.max(img)
    img = np.uint8((img - mmin) * 255.0 / (mmax - mmin))
    r, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    mts = cv2.moments(img, True)
    x, y = mts['m10'] / mts['m00'], mts['m01']/mts['m00']
    dx = img.shape[1] / 2.0 - x
    dy = img.shape[0] / 2.0 - y
    trf = np.float32([[1, 0, dx, 0, 1, dy]]).reshape(2, 3)
    img = cv2.warpAffine(img, trf, img.shape)

    img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.circle(img2, (int(x),int(y)), 5, (0, 0, 255))
    cv2.imshow("asd", img2)
    cv2.waitKey()


if __name__ == '__main__':
    test()
    exit()

    img = cv2.imread("""D:\dokumentumok\opencv\sources\samples\data\digits.png""")
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
    ret,result,neighbours,dist = knn.find_nearest(test,k=5)

    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print accuracy