import cv2
import numpy as np
from OCR import *
from pprint import pprint
from glob import glob
import os


def convert():
    for i in range(10):
        print i
        folder = "%d/" % i
        files = glob(folder + "*.png")
        for file in files:
            fname = os.path.basename(file)
            fname2 = fname.rjust(8, "0")
            img = cv2.imread(file, 0)
            # print fname
            img2 = 255 - extract_digit(img, True, True, True)
            cv2.imwrite(folder + fname2, img2)

def test_knn(knn, fname, digit, k):
    img = cv2.imread(fname, 0)
    img = extract_digit(img)
    img = 255 - img
    img2 = np.float32(img.reshape((1, 400)))
    ret, result, neighbours, dist = knn.find_nearest(img2, k=k)

    return int(result[0,0])

if __name__ == '__main__':
    x = np.zeros((10, 224, 20, 20))

    for i in range(10):
        print i
        folder = "%d/" % i
        files = glob(folder + "*.png")
        for file in files:
            fname = os.path.basename(file)
            idx = int(fname[:4])
            img = cv2.imread(file, 0)
            x[i, idx, :, :] = img


    # Make it into a Numpy array. It size will be (50,100,20,20)
    # x = np.array(cells)

    # Now we prepare train_data and test_data.
    train = x[:,:112].reshape(-1,400).astype(np.float32) # Size = (1120,400)
    test = x[:,112:224].reshape(-1,400).astype(np.float32) # Size = (1120,400)

    # Create labels for train and test data
    k = np.arange(10)
    train_labels = np.repeat(k,112)[:,np.newaxis]
    test_labels = train_labels.copy()

    # Initiate kNN, train the data, then test it with test data for k=1
    knn = cv2.KNearest()
    knn.train(train,train_labels)
    print test.shape
    for i in range(1, 10):
        print "----- k=%d" % i
        ret,result,neighbours,dist = knn.find_nearest(test,k=i)

        # Now we check the accuracy of classification
        # For that, compare the result with test_labels and check which are wrong
        matches = result==test_labels
        correct = np.count_nonzero(matches)
        accuracy = correct*100.0/result.size
        print accuracy
        print correct, result.size


    img = cv2.imread("""../9_2.png""", 0)
    img = extract_digit(img)
    img = 255-img
    # cv2.imshow("i", img)
    # cv2.waitKey()
    img = np.float32(img.reshape((1, 400)))
    ret,result,neighbours,dist = knn.find_nearest(img,k=5)
    print result

    print "--------------------------"
    for k in range(1, 10):
        print "------ k=%d" % k
        files = glob("""../digit_tests/*.png""")
        for f in files:
            i = int(os.path.basename(f)[0])

            res = test_knn(knn, f, i, k)
            if res != i:
                print res, i

