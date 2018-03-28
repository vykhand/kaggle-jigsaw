import kaggle_jigsaw.util as u

import logging
log = logging.getLogger("jigsaw")

datadir = u.datadir
t = u.get_timer()

import kaggle_jigsaw.constants as k

from sklearn.base import TransformerMixin
import langdetect
from translation import bing
import string

import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

import pandas as pd
import numpy as np

import json

class TextPreprocessor(TransformerMixin):
    def __init__(self, **kwargs):
        self._dedup_thresh = kwargs.get("deduplicate_threshold", 0)
        self._clean_stopwords = kwargs.get("clean_stopwords")
        self._remove_appo = kwargs.get("remove_apostrophes")
        self._repl_smileys = kwargs.get("repl_smileys")
        self._translate = kwargs.get("translate")
        self._tokenizer = TweetTokenizer()
        self._lem  = WordNetLemmatizer()

        # clean stopwords
        if self._clean_stopwords:
            try:
                self.eng_stopwords =  set(stopwords.words("english"))
            except LookupError as e:
                log.warn("NLTK stopwords not found. Downloading.")
                nltk.download("stopwords")
                eng_stopwords =  set(stopwords.words("english"))
        if self._remove_appostrophes:
            with open(u.get_file(k.APOSTROPHES_FILE), 'r') as f:
                self._APPO = json.load(f)


    def _dedup(self, text):
        word_list = text.split()
        num_words = len(word_list)
        if num_words > 0:
            num_unique_words = len(set(word_list))
            unique_ratio = num_words / num_unique_words
            if unique_ratio > self._dedup_thresh:
                text = " ".join(text.split())[:num_unique_words]
        return text

    def _remove_stopwords(self, text):
        words = self._tokenizer.tokenize(text)
        words = [self._lem.lemmatize(word, "v") for word in words]
        text = " ". join([w for w in text.lower().split() if w not in self.eng_stopwords])
        return text

    def _remove_appostrophes(self, text):
        words = self._tokenizer.tokenize(text)
        words = [self._APPO[word] if word in self._APPO else word for word in words]

        text = " ".join(words)
        return text

    def _translate_to_eng(self, text):
        try:
          lang = langdetect.detect(text)
        except Exception:
            lang = 'en'

        if lang != 'en':
            try:
                text = bing(text, dst='en')
            except Exception:
                pass
        return text
    def _common_transforms(self, X):
        # replace urls
        re_url = re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\
                    .([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
                    re.MULTILINE|re.UNICODE)
        # replace ips
        re_ip = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
        # username
        re_username = re.compile("\[\[.*\]")

        # this is non-optional preprocessing section
        # replace URLs
        X = re_url.sub("URL", X)
        # replace IPs
        X = re_ip.sub("IPADDRESS", X)
        # replace usernames
        X = re_username.sub("USERNAME", X)
        #remove punctuation
        X = re.sub(r'[^\w\s]', ' ', X)
        # remove newline
        X = X.replace('\n', ' ')
        X = X.replace('\n\n', ' ')

        #to lower
        X = X.lower()
        # remove multiple spaces
        X  = ' '.join(X.split())

        #remove numbers
        X = re.sub("\d+", "", X)

        # remove repetitive letters
        X =  re.sub(r"(\w)\1{2,}", r"\1", X)

        #remove repetitive words
        X = re.sub(r'\b(\w+)( \1\b)+', r"\1", X)

        # some custom replacements
        # TODO: make more of those and load them from special file

        X = re.sub(r"\b(f u c k)\b", r"fuck", X)
        X = re.sub(r"\b(f uu c kk)\b","fuck", X)
        X = re.sub(r"\b(f u)\b","fuck", X)


        return X

    def _replace_smileys(self, X):
        smiley_patterns = [     (':d',  ' smile '),
                                (':dd', ' smile '),
                                (':p',  ' smile '),
                                ('8\)', ' smile '),
                                (':-\)', ' smile '),
                                (':\)',  ' smile '),
                                (';\)',  ' smile '),
                                ('\(-:', ' smile '),
                                ('\(:',  ' smile '),
                                ('yay!', ' good '),
                                ('yay', ' good '),
                                ('yaay', ' good '),
                                (':/', ' worry '),
                                (':&gt;', ' angry '),
                                (":'\)", ' sad '),
                                (':-\(', ' sad '),
                                (':\(', ' sad '),
                                (':s',  ' sad '),
                                (':-s', ' sad '),]
        patterns = [(re.compile(regex), repl) for (regex, repl) in smiley_patterns]
        ret = X
        for (pattern, repl) in patterns:
            ret = pattern.sub(repl, ret)
        return ret

    def transform(self, X):
        X = pd.Series(X)
        transforms  =  [("common", self._common_transforms)]
        if self._dedup_thresh > 0: transforms.append(("dedup",self._dedup))
        if self._clean_stopwords: transforms.append(("stopwords", self._remove_stopwords))
        if self._remove_appo: transforms.append(("appostrophes",self._remove_appostrophes))
        if self._translate: transforms.append(("translate",self._translate_to_eng))
        if self._repl_smileys: transforms.append(("remove smileys", self._replace_smileys))

        for trans in transforms:
            t.tic("Running text transformation: " + trans[0])
            X = X.apply(trans[1])
            t.toc()

        return X.values




