#import constants as C

from time import time
import numpy as np
import pandas as pd
import os

import kaggle_jigsaw.util as u

from scipy.sparse import csr_matrix
import scipy
from sklearn.pipeline import Pipeline

import logging
log = logging.getLogger("jigsaw")

datadir = u.datadir
t = u.get_timer()

import kaggle_jigsaw.constants as k
import hashlib

import langdetect
from translation import bing
import string

import re
from nltk.corpus import stopwords
import nltk

from .text import TextPreprocessor

class BasicPreprocessor:
    def __init__(self, exec_params, feature_params):
        self._exec_par = exec_params
        self._feat_par = feature_params
        self._prepr_par = feature_params.get("preprocessing_params", {})
        self.disable_cache = exec_params["disable_cache"]
        ## generating md5 for the given param combination
        self.set_cached_file_names()
        # TODO: add fit on all option
        #self.fit_on_all = feature_params["fit_on_all"]
        self._train_file = feature_params.get("train_file", k.TRAIN_FILE)
        self._test_file = feature_params.get("test_file", k.TEST_FILE)

        log.debug("cached_name: {}".format(self.cached_file_names))
    def set_cached_file_names(self, typ = "csr_matrix"):
        s = u.dict_to_md5(self._feat_par)
        n = self._exec_par["cached_prefix"] + "_{}".format(s)
        if typ == "csr_matrix":
            self.cached_file_names = ["X_train_{}.npz".format(n), "X_test_{}.npz".format(n)]
        else:
            self.cached_file_names = [n + ".h5"]
        return self

    def blob_is_available(self):
        return all(u.is_available(f) for f in self.cached_file_names)

    def read_preprocessed(self):
        prepr = TextPreprocessor(**self._prepr_par)
        X_train, X_test  = self.read_train_test()
        X_train = prepr.transform(X_train)
        X_test = prepr.transform(X_test)

        return X_train, X_test

    def read_train_test(self):
        t.tic("Read train and test csv files")
        train = pd.read_csv(u.get_file( self._train_file))
        test = pd.read_csv(u.get_file( self._test_file))
        t.toc()
        return train[k.COMMENT], test[k.COMMENT]

    def get_y(self):
        t.tic("Read target variable")
        self.y_train =  pd.read_csv(u.get_file( k.TRAIN_FILE)).iloc[:, 2:]
        t.toc()
    def read_from_cache(self, typ = "csr_matrix"):
        '''
        @param typ: ["csr_matrix", "pandas", "numpy"]
        #TODO: this is in fact just uses "csr_matrix" mode
        #TODO: refactor
        Basic procedure that reads file from Azure storage and locally
        sets X_train
        '''
        if typ == "pandas":
            fpath = u.get_file(self.cached_file_names[0])

            self.X_train = pd.read_hdf(fpath, "X_train")
            self.X_test = pd.read_hdf(fpath, "X_test")
        elif typ == "csr_matrix":
            # np.savez appends .npz at the end
            train_fname = self.cached_file_names[0]
            test_fname = self.cached_file_names[1]
            self.X_train = u.load_sparse_csr(u.get_file(train_fname))
            self.X_test = u.load_sparse_csr(u.get_file(test_fname))
        #TODO: add some handling for numpy
        else:
            raise ValueError("Unknown type")
        return self.X_train, self.X_test

    def get_pipeline(self):
        self.pipeline = Pipeline([])
        return self.pipeline

    def save_to_cache(self):
        '''
        cache a file to datadir folder and if not to blob
        '''
        if isinstance(self.X_train, pd.core.frame.DataFrame):
            fname = self.cached_file_names[0]
            self.X_train.to_hdf(os.path.join(datadir, fname), "X_train", mode = "w")
            self.X_test.to_hdf(os.path.join(datadir, fname), "X_test", mode =  "r+")

            u.put_file_to_blob(fname)
        elif isinstance(self.X_train, scipy.sparse.csr.csr_matrix):
            train_fname = os.path.join(datadir, self.cached_file_names[0])
            test_fname = os.path.join(datadir,  self.cached_file_names[1])

            u.save_sparse_csr(train_fname, self.X_train)
            u.save_sparse_csr(test_fname, self.X_test)
            u.put_file_to_blob(train_fname)
            u.put_file_to_blob(test_fname)
        else:
            raise ValueError("don't know how to save type {} to cache".format(type(self.X_train)))

    def get_features(self):
        '''
        runs preprocessing. As a result you get X_train, X_test, y_train attributes
        '''
        self.get_y()

        if not self.disable_cache and self.blob_is_available():
            t.tic("Reading features from cache {}".format(self.cached_file_names))
            self.X_train, self.X_test =  self.read_from_cache()
            t.toc()
        else:
            t.tic("Run preprocessing")
            self.get_pipeline()
            X_train, X_test = self.read_preprocessed()
            self.X_train = self.pipeline.fit_transform(X_train, self.y_train)
            self.X_test = self.pipeline.transform(X_test)
            t.toc()

            if not self.disable_cache:
                t.tic("Saving cache locally and to Azure")
                self.save_to_cache()
                t.toc()
        return self.X_train, self.X_test, self.y_train




