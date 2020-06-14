import numpy as np
from collections import defaultdict


class KNNClassifier():
    """
    KNN Classifier.

    Parameters
    ------------------
    K: int
        K nearest neighbors. default 1 to m(Sample Size)
    distance: string
        distance metrics
    weight: (Not Using)

    Attributes
    ------------------
    """

    def __init__(self, algorithm='brute', distance='Euclidean', K=1, p=1):

        self.K = K

        self.algorithm = algorithm
        """
        3 algorithms
        """

        self.distance = distance

        # When using _Minkowski_Distance, p needs to be set
        self.p = p

        if self.distance == 'Euclidean':
            self.distance_metrics = self._Euclidean_Distance
        elif self.distance == 'Manhattan':
            self.distance_metrics = self._Manhattan_Distance
        elif self.distance == 'Minkowski':
            self.distance_metrics = self._Minkowski_Distance
        elif self.distance == 'Mahalanobis':
            self.distance_metrics = self._Mahalanobis_Distance
        elif self.distance == 'Haversine':
            self.distance_metrics = self._Haversine_Distance
        elif self.distance == 'Cosine':
            self.distance_metrics = self._Cosine_Similarity

    def _Euclidean_Distance(self, X_test, X_train):
        m, n = X_train.shape
        # .tile(A,reps) repeat A for 'reps' times
        diff = np.tile(X_test, (m, 1)) - X_train
        sqDiff = diff ** 2
        sumSqDiff = sqDiff.sum(axis=1)
        dist = sumSqDiff ** 0.5
        # .argsort() return the indices of sorted values
        return dist.argsort()

    def _Manhattan_Distance(self, X_test, X_train):
        m, n = X_train.shape
        diff = np.tile(X_test, (m, 1)) - X_train
        absDiff = np.abs(diff)
        dist = absDiff.sum(axis=1)
        return dist.argsort()

    def _Minkowski_Distance(self, X_test, X_train):
        m, n = X_train.shape
        diff = np.tile(X_test, (m, 1)) - X_train
        absDiff = np.abs(diff)
        sqAbsDiff = absDiff ** self.p
        totalDiff = sqAbsDiff.sum(axis=1)
        dist = totalDiff ** (1 / self.p)
        return dist.argsort()

    def _Mahalanobis_Distance(self, X_test, X_train):
        pass

    def _Haversine_Distance(self, X_test, X_train):
        pass

    def _Cosine_Similarity(self, X_test, X_train):
        m, n = X_train.shape
        X_mat = np.tile(X_test, (m, 1))
        dotproduct = np.sum(X_mat * X_train, axis=1)
        cos = dotproduct / np.sqrt((np.sum(X_mat * X_mat, axis=1) * np.sum(X_train * X_train, axis=1)))
        return cos.argsort()

    def fit(self, X_train, y_train, verbose=True):
        """
        Fit method for training data.

        Parameters:
        -----------------------
        X_train: {array-like}, shape = [n_samples, n_features]
            Training matrix, where 'n_samples' is the number of samples
            and 'n_features' is the number of features
        y_train: {array-like}, shape = [n_samples]
            Target labels

        Attributes:
        -----------------------
        d_record_: list
            Record all distance.
        error_rate_: list
            Record all missclassification rate.

        Returns:
        ------------------------
        self: object

        """
        # Check datatype
        if isinstance(X_train, np.ndarray) and isinstance(y_train, np.ndarray):
            pass
        else:
            try:
                X_train = np.array(X_train)
                y_train = np.array(y_train)
            except:
                raise TypeError('numpy.array required for input data')

        if self.algorithm == 'brute':
            self.X_train = X_train
            self.y_train = y_train

    def _pred(self, X_test_row):

        if isinstance(X_test_row, np.ndarray):
            pass
        else:
            try:
                X_test_row = np.array(X_test_row)
            except:
                raise TypeError('numpy.array required for input data')

        SortedDistIndex = self.distance_metrics(X_test_row, self.X_train)

        classCount = defaultdict(int)
        ans = []

        for j in range(self.K):
            label = self.y_train[SortedDistIndex[j]][0]
            classCount[label] += 1

        sortedPredict = sorted(classCount.items(), key=lambda item: item[1])
        ans.append(sortedPredict[0][0])

        return ans

    def predict(self, X_test):

        if isinstance(X_test, np.ndarray):
            pass
        else:
            try:
                X_test = np.array(X_test)
            except:
                raise TypeError('numpy.array required for input data')

        m, n = X_test.shape

        prediction = []

        for i in range(m):
            ans = self._pred(X_test[i])
            prediction.append(ans)

        return prediction


if __name__ == '__main__':
    from sklearn import datasets
    import pandas as pd

    iris = datasets.load_iris()

    feature = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = pd.DataFrame(iris.target, columns=['target'])
    df = pd.concat([feature, target], axis=1)
    df = df[df['target'] != 2]
    df = df.reset_index()
    target = df['target'].to_frame()
    feature = df.drop('target', axis=1)
    X_train = feature.to_numpy()
    # data_mat = np.asmatrix(feature)
    y_train = target.to_numpy()
    # label_mat = np.asmatrix(target)
    knn = KNNClassifier(distance='Euclidean', p=1, K=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)

    from sklearn.metrics import confusion_matrix

    print(confusion_matrix(y_train, y_pred))
