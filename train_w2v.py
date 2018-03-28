import os, sys
import pandas as pd
import numpy as np
import csv

import kaggle_jigsaw.util as u

import logging
log = logging.getLogger("jigsaw")

datadir = u.datadir
t = u.get_timer()
u.set_console_logger(log)

from gensim.models import KeyedVectors, Word2Vec
from kaggle_jigsaw.preprocessing.basic import BasicPreprocessor
import yaml

from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from string import punctuation, ascii_lowercase
import regex as re
tokenizer = WordPunctTokenizer()

vocab = Counter()

def text_to_wordlist(text):
    # Tokenize
    text = tokenizer.tokenize(text)
    # Return a list of words
    vocab.update(text)
    return text

def process_comments(list_sentences):
    comments = []
    for text in list_sentences:
        txt = text_to_wordlist(text)
        comments.append(txt)
    return comments

t.tic("tokenization")
embed_size  = 300
conf = yaml.load(open("conf_template/GRU.yaml"))
prepr = BasicPreprocessor(conf["exec_params"], conf["feature_params"])
Xtr, Xte = prepr.read_preprocessed()
comments = process_comments(list(Xtr) + list(Xte))
t.toc()

t.tic("training word2vec with embedding size {}".format(embed_size))
model_fname = "gensim_word2vec_{}d.vec".format(embed_size)
model_fpath = os.path.join(u.datadir, model_fname)
model = Word2Vec(comments, size=embed_size, window=5, min_count=5, workers=4, sg=0, negative=5)
model.wv.save_word2vec_format(model_fpath, binary=True)
t.toc()

t.tic("uploading to azure")
u.put_file_to_blob(model_fname)
t.toc()