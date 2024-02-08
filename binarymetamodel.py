import numpy as np
from sklearn.base import BaseEstimator


class BinaryMetaModel(BaseEstimator):
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def fit(self, X, y):
        self.model1.fit(X, y)
        pred = self.model1.predict(X).reshape((X.shape[0], -1)).astype(int)
        X2 = np.hstack((X, pred))
        y2 = pred & y.reshape((X.shape[0], -1)).astype(int)
        self.model2.fit(X2, y2)
        return self

    def predict(self, X):
        pred = self.model1.predict(X).reshape((X.shape[0], -1)).astype(int)
        X2 = np.hstack((X, pred))
        return self.model2.predict(X2)
