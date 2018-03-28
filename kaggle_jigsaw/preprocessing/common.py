import logging
log = logging.getLogger("jigsaw")

from .basic import BasicPreprocessor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import re, string
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class WordCharPreprocessor(BasicPreprocessor):

    def get_pipeline(self):
        steps = []
        re_tok = re.compile('([{}“”¨«»®´·º½¾¿¡§£₤‘’])'.format(string.punctuation))
        tokenize = lambda s: re_tok.sub(r' \1 ', s).split()
        steps.append( ("word_vec", TfidfVectorizer( tokenizer=tokenize,
                                                   analyzer = "word",
                                            **self._feat_par["wordvec_params"])))
        steps.append(("char_vec", TfidfVectorizer( tokenizer=tokenize, analyzer = "char",
                           **self._feat_par["charvec_params"])))

        self.pipeline = FeatureUnion(steps)

        return self.pipeline

