from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy import sparse
import pandas as pd
import numpy as np
import kaggle_jigsaw.util as u
import logging
log = logging.getLogger("jigsaw")
t = u.get_timer()

class BaseSklearnEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, model_params):
        self._estimator = estimator
        self._model_params = model_params

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_clf'])
        preds = np.zeros((x.shape[0],len(self._clf.keys())))
        for k in self._clf.keys():
            preds[:,k] = self._clf[k].predict(x)
        return preds

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_clf'])
        preds = np.zeros((x.shape[0],len(self._clf.keys())))
        for k in self._clf.keys():
            preds[:,k] = self._clf[k].predict_proba(x)[:,1]
        return preds

    def fit(self, x, y, xval = None, yval = None):
        # Check that X and y have correct shape
        nlabs = y.shape[1]
        x, y = check_X_y(x, y, accept_sparse=True, multi_output=True)
        self._clf = {}
        for i in range(0, nlabs):
            t.tic("Fitting on label: {}".format(i))
            self._clf[i] = self._estimator(**self._model_params).fit(x, y[:,i])
            #self._clf[i] = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y[:,i])
            t.toc()
        return self


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_params):
        self.C = model_params["C"]
        self.dual = model_params["dual"]
        self.n_jobs = model_params["n_jobs"]

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        preds = np.zeros((x.shape[0],len(self._clf.keys())))
        for k in self._clf.keys():
            preds[:,k] = self._clf[k].predict(x.multiply(self._r[k]))
        return preds

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        preds = np.zeros((x.shape[0],len(self._clf.keys())))
        for k in self._clf.keys():
            preds[:,k] = self._clf[k].predict_proba(x.multiply(self._r[k]))[:,1]
        return preds

    def pr(self, x, y_i, y):
        p = x[y==y_i].sum(0)
        return (p+1) / ((y==y_i).sum()+1)

    def fit(self, x, y, xval = None, yval = None):
        # Check that X and y have correct shape
        nlabs = y.shape[1]
        x, y = check_X_y(x, y, accept_sparse=True, multi_output=True)
        self._r = {}
        self._clf = {}
        for i in range(0, nlabs):
            t.tic("Fitting on label: {}".format(i))
            self._r[i] = sparse.csr_matrix(np.log(self.pr(x,1,y[:,i]) / self.pr(x,0,y[:,i])))
            x_nb = x.multiply(self._r[i])
            #self._clf[i] = SVC(C=self.C).fit(x_nb, y[:,i])
            self._clf[i] = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y[:,i])
            t.toc()
        return self


