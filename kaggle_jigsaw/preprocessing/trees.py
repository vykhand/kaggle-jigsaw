
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from time import time
import re
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import string
import os
import kaggle_jigsaw.util as u
import kaggle_jigsaw.constants  as C
from . common import WordCharPreprocessor

from sklearn.pipeline import FeatureUnion

import logging
log = logging.getLogger("jigsaw")

datadir = u.datadir
t  =  u.get_timer()

class CountsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, use_cache=False):

        self.start_time = time()

       # self.eng_stopwords =  set(stopwords.words("english"))

        with open(u.get_file( C.BADWORDS_FNAME),"r", encoding="latin1") as f:
            self.bad_words = set([s.strip() for s in f.read().split("\n")])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transforms = {
            "count_caps": lambda x: sum(map(str.isupper, x)),
            "count_bang": lambda x: len([c for c in str(x) if c == "!"]),
            "count_Q": lambda x: len([c for c in str(x) if c == "?"]),
            "count_sent": lambda x: len(re.findall("\n",str(x)))+1,
            "count_word": lambda x: len(str(x).split()),
            "count_unique_word": lambda x: len(set(str(x).split())),
            "count_letters": lambda x: len(str(x)),
            "count_punctuations": lambda x: len([c for c in str(x) if c in string.punctuation]),
            "count_ascii": lambda x: len([c for c in str(x) if c in string.printable]),
            "count_non_ascii": lambda x: len([c for c in str(x) if c not in string.printable]),
            "count_words_upper": lambda x: len([w for w in str(x).split() if w.isupper()]),
            "count_words_title": lambda x: len([w for w in str(x).split() if w.istitle()]),
         #   "count_stopwords": lambda x: len([w for w in str(x).lower().split() if w in self.eng_stopwords]),
            "mean_word_len": lambda x: np.mean([len(w) for w in str(x).split()]),
            "count_badwords": lambda x: len([w for w in str(x).lower().split() if w in self.bad_words])
        }

        df = pd.DataFrame(index = X.index)

        for k, v in transforms.items():
            t = time()
            df[k] = X.apply(v)
            log.info("processed feature: {} : time {} : total_time {}".format(k,
                                                            time() -t,
                                                            time() - self.start_time ))
        return df

class CntWordCharPreprocessor(WordCharPreprocessor):
    def get_pipeline(self):
        pipe = super().get_pipeline()
        return FeatureUnion(transformer_list =
                                [("wordchar", pipe),
                                ("counts",Pipeline(steps =[ ("cnts", CountsTransformer()),
                                                            ("scaler", MaxAbsScaler())]))])
