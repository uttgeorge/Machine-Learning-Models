import numpy as np
import KNNClassifier

import matplotlib.pyplot as plt
if __name__ == '__main__':
    image_size = 28 # width and length
    no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size
    data_path = "/mnist/"
    train_data = np.loadtxt(data_path + "mnist_train.csv",
                            delimiter=",")
    test_data = np.loadtxt(data_path + "mnist_test.csv",
                           delimiter=",")
    fac = 0.99 / 255
    train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
    test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])

    knn = KNNClassifier(distance='Euclidean', K=5)
    knn.fit(train_imgs,train_labels)
    results = knn.predict(test_imgs)

    #
    # for i in range(10):
    #     img = train_imgs[i].reshape((28,28))
    #     plt.imshow(img, cmap="Greys")
    #     plt.show()

    from sklearn.metrics import confusion_matrix
    confusion_matrix(test_labels,results)