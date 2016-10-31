import numpy as np
import cv2
from OCR import *

def test():
    img = cv2.imread("""D:/dokumentumok/opencv/sources/samples/python2/data/digits.png""", 0)
    img2 = img
    for i in range(50):
        for j in range(100):
            pic = img[i * 20 : (i+1) * 20, j * 20 : (j+1) * 20]
            pic = 255 - extract_digit(pic, True, True, True)
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