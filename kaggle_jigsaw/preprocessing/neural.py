from .basic import BasicPreprocessor
import kaggle_jigsaw.util as u
import numpy as np
import pandas as pd
import os
import h5py
import csv
import kaggle_jigsaw.util as u

import logging
log = logging.getLogger("jigsaw")

datadir = u.datadir
t = u.get_timer()

from keras.preprocessing import text, sequence
import warnings
warnings.filterwarnings('ignore')
import pickle

#from sklearn.base import BaseEstimator, TransformerMixin

#class KerasTransformer(BaseEstimator, TransformerMixin):
#    def transform(self,X)

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

class KerasWithEmbeddings(BasicPreprocessor):
    '''
    Preprocessor for Keras that uses pre-trained embeddings
    Heavily redefines BasicPreprocessor
    '''
    def __init__(self, embedding_transformer, exec_params, feature_params):
        super().__init__(exec_params, feature_params)
        self._maxlen  = feature_params["maxlen"]
        self._max_features = feature_params["max_features"]
        self._embedding_file = feature_params.get("embedding_file")
        self._embed_size = feature_params["embed_size"]
        if self._embedding_file is not None:
            self._emb_transformer = embedding_transformer(self._embedding_file,
                                                          self._max_features,
                                                          self._embed_size)

    def set_cached_file_names(self):
        ''' in case of neural nets everything is in one h5 file'''
        s = u.dict_to_md5(self._feat_par)
        self.cached_file_names =  [self._exec_par["cached_prefix"] + "_{}.h5".format(s),
                                   self._exec_par["cached_prefix"] + "_{}_tokenizer.pkl".format(s)]

    def transform_train_test(self, xtrain, xtest):
        tokenizer = text.Tokenizer(num_words=self._max_features)
        tokenizer.fit_on_texts(list(xtrain) + list(xtest))
        X_train = tokenizer.texts_to_sequences(xtrain)
        X_test = tokenizer.texts_to_sequences(xtest)
        X_train = sequence.pad_sequences(X_train, maxlen=self._maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=self._maxlen)

        self._tokenizer  = tokenizer
        return X_train, X_test

    def read_from_cache(self):
        '''Reads X_train, X_test, and embedding matrix from an h5 file'''
        cached_path = u.get_file(self.cached_file_names[0])
        with h5py.File(cached_path, "r") as f:
            X_train = f["X_train"][:]
            X_test = f["X_test"][:]
            if self._embedding_file is not None:
                embedding_matrix = f["embedding_matrix"][:]
            else: embedding_matrix = None
        return X_train, X_test, embedding_matrix

    def save_to_cache(self):
        f = self.cached_file_names[0]
        with h5py.File(os.path.join(datadir, f), "w") as h5:
            h5.create_dataset("X_train", data = self.X_train)
            h5.create_dataset("X_test", data = self.X_test)
            if self._embedding_file is not None:
                h5.create_dataset("embedding_matrix", data = self._embedding_matrix)
        u.put_file_to_blob(f)
        # saving tokenizer
        tok_fname = self.cached_file_names[1]
        with open(os.path.join(datadir, tok_fname), 'wb') as handle:
            pickle.dump(self._tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        u.put_file_to_blob(tok_fname)


    def get_pipeline(self):
        return NotImplementedError

    def get_features(self):
        self.get_y()

        if not self.disable_cache and self.blob_is_available():
            t.tic("Reading features from cache {}".format(self.cached_file_names))
            self.X_train, self.X_test, self._embedding_matrix =  self.read_from_cache()
            t.toc()
        else:
            t.tic("Run preprocessing")
            self.X_train, self.X_test  = self.read_preprocessed()
            self.X_train, self.X_test  = self.transform_train_test(self.X_train,
                                                                   self.X_test)
            if self._embedding_file is not None:
                self._embedding_matrix = self._emb_transformer._get_embedding_matrix(self._tokenizer)
            else: self._embedding_matrix = None
            t.toc()

            if not self.disable_cache:
                t.tic("Saving cache locally and to Azure")
                self.save_to_cache()
                t.toc()

        return self.X_train, self.X_test, self.y_train, self._embedding_matrix

