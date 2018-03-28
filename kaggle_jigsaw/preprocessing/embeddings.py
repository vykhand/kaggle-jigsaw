from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
import csv

import kaggle_jigsaw.util as u

import logging
log = logging.getLogger("jigsaw")

datadir = u.datadir
t = u.get_timer()

from gensim.models import KeyedVectors

## the code on this page is heavily borrowed from here :

## https://github.com/neptune-ml/kaggle-toxic-starter

class EmbeddingsMatrix(TransformerMixin):
    def __init__(self, embedding_file, max_features, embedding_size):
        self._embedding_file = embedding_file
        self._max_features  = max_features
        self._embed_size = embedding_size
    def fit(self, tokenizer):
        self.embedding_matrix = self._get_embedding_matrix(tokenizer)
        return self
    def transform(self, tokenizer):
        return {'embeddings_matrix': self.embedding_matrix}
    def _get_embedding_matrix(self, tokenizer):
        return NotImplementedError



class GloveEmbeddingMatrix(EmbeddingsMatrix):
    def _get_embedding_matrix(self, tokenizer):

        t.tic("Fetching and reading embedding file: " + self._embedding_file )

        embeddings = pd.read_table(u.get_file(self._embedding_file),
                         sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

        t.toc()

        emb_mean, emb_std = np.mean(embeddings.values), np.std(embeddings.values)

        word_index = tokenizer.word_index
        nb_words = min(self._max_features, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, self._embed_size))
        for word, i in word_index.items():
            if i >= self._max_features: continue
            embedding_vector = embeddings.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector.as_matrix()
        return embedding_matrix


class Word2VecEmbeddingsMatrix(EmbeddingsMatrix):
    def _get_embedding_matrix(self, tokenizer):
        #model = KeyedVectors.load_word2vec_format(u.get_file(self._embedding_file), binary=True)
        model = KeyedVectors.load_word2vec_format(u.get_file(self._embedding_file), binary=False)
        emb_mean, emb_std = model.syn0.mean(), model.syn0.std()

        word_index = tokenizer.word_index
        nb_words = min(self._max_features, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, self._embed_size))
        for word, i in word_index.items():
            if i >= self._max_features:
                continue
            try:
                embedding_vector = model[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                continue
        return embedding_matrix


class FastTextEmbeddingsMatrix(EmbeddingsMatrix):
    def _get_embedding_matrix(self, tokenizer):
        embeddings_index = dict()
        with open(u.get_file(self._embedding_file),  encoding="latin1") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if i == 0:
                    continue
                values = line.split(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                if coefs.shape[0] != self._embed_size:
                    continue
                embeddings_index[word] = coefs

        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()

        word_index = tokenizer.word_index
        nb_words = min(self._max_features, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, self._embed_size))
        for word, i in word_index.items():
            if i >= self._max_features:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
