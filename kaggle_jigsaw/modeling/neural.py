import os
import pandas as pd
import numpy as np
#import kaggle_jigsaw.util as u
from kaggle_jigsaw import util as u
import logging
log = logging.getLogger("jigsaw")
t = u.get_timer()
from .basic import BasicCrossValidator

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
import kaggle_jigsaw.constants as C
from sklearn.metrics import roc_auc_score

from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Flatten, Dropout
from keras.layers import GRU, CuDNNGRU, CuDNNLSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPool1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop

import uuid
import yaml

from keras import backend as K

cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.best_epoch = 1
        self.best_auc = 0
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            logs['roc_auc']  = score
            if score > self.best_auc:
                self.best_epoch = epoch
                self.best_auc = score
            log.info("epoch: %d - AUC: %.6f" % (epoch+1, score))
        return

class KerasCrossValidator(BasicCrossValidator):
    def run_cv(self):
        self._bestmodel_fnames = {}
        self._best_epochs = {}
        score = super().run_cv()
        return score

    def run_cv_iter(self,  x, y, xval, yval, val_ind,  fold_name):
        model, score = super().run_cv_iter(x, y, xval, yval, val_ind, fold_name)
        if hasattr(model, "_bestmodel_fname"):
            self._bestmodel_fnames[fold_name] = model._bestmodel_fname
        if hasattr(model, "_best_epoch"):
                self._best_epochs[fold_name] = model._best_epoch
        #clearing GPU memory after each iteration
        K.clear_session()
        return model, score

    def _fit(self, X, y, xval=None, yval=None):
        # if running after CV
        mean_epoch = None
        if (xval is None or yval is None) and \
            len(self._best_epochs) > 0:
            mean_epoch = int(np.mean(list(self._best_epochs.values())))
        return self._model.fit(X, y, xval, yval, mean_epoch)

    def save_outputs(self):
        super().save_outputs()
        with open(C.OUTPUT_DIR + "/" + self._file_prefix + \
            "_bestmodel_fnames.yml", "w") as f:
            yaml.dump(self._bestmodel_fnames, f, default_flow_style=False)
        with open(C.OUTPUT_DIR + "/" + self._file_prefix + \
            "_bestrpochs.yml", "w") as f:
            yaml.dump(self._best_epochs, f, default_flow_style=False)


class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, embedding_matrix, feature_params, model_params, fit_params):
        self._embedding_matrix = embedding_matrix
        self._feat_par = feature_params
        self._fit_par = fit_params
        self._mod_par  = model_params
        self._maxlen  = feature_params["maxlen"]
        self._max_features = feature_params["max_features"]
        self._embed_size = feature_params["embed_size"]
        self._pretrained_embeddings = model_params["pretrained_embeddings"]

    def _get_model(self):
        raise NotImplementedError
        return self

    def _get_embedding_layer(self):
        if self._pretrained_embeddings:
            x = Embedding(self._max_features, self._embed_size, weights=[self._embedding_matrix], trainable=False)
        else:
            x = Embedding(self._max_features, self._embed_size)
        return x
    def predict(self, x):
        return self.predict_proba(x)

    def predict_proba(self, x):
        check_is_fitted(self, ["_clf"])
        return self._clf.predict(x)

    def fit(self, x, y, xval = None, yval = None, mean_epoch = None):
        model =  self._get_model()
        self._uid = uuid.uuid4()
        l_y = y.values if isinstance(y, pd.DataFrame) else y
        callbks = []

        use_tensorboard = self._mod_par.get("use_tensorboard")
        if use_tensorboard:
            logdir = os.path.join(u.datadir, "logdir.{}".format(self._uid))
            log.info("TensorBoard logging directory: {}".format(logdir))
            os.makedirs(logdir, exist_ok=True)
            callbks.append(TensorBoard(log_dir = logdir,
                            histogram_freq = self._mod_par.get("tensorboard_hist_freq", 0),
                            write_graph = self._mod_par.get("tensorboard_write_graph", False),
                            write_grads = self._mod_par.get("tensorboard_write_grads", False),
                            embeddings_freq = self._mod_par.get("tensorboard_embeddings_freq", 0),
                            batch_size  = self._fit_par.get("batch_size", 32)))
            self._tensorboard_logdir = logdir

        if xval is not None and yval is not None:
            RocAuc = RocAucEvaluation(validation_data=(xval, yval), interval=1)
            callbks.append(RocAuc)
            patience = self._mod_par.get("early_stop_patience", 0 )
            log.debug("Early stopping patience: {}".format(patience))
            if patience > 0:
                self._bestmodel_fname = "bestmodel_{}.hdf5".format(self._uid)
                bestmodel_fpath = C.OUTPUT_DIR + "/" + self._bestmodel_fname
                callbks.append(EarlyStopping(monitor = "roc_auc", patience = patience,
                                mode="max", verbose=1 ))
                callbks.append(ModelCheckpoint(monitor = "roc_auc", mode="max" , filepath = bestmodel_fpath,
                                                verbose = 1, save_best_only=True))

            model.fit(x, l_y, validation_data = (xval, yval),
                               callbacks = callbks,
                               **self._fit_par)
            self._best_epoch = RocAuc.best_epoch
        else:
            if mean_epoch is not None:
                log.debug("Using average best epoch for retraining: {}".format(mean_epoch))
                self._fit_par["epochs"] = mean_epoch
            model.fit(x, l_y, callbacks = callbks, **self._fit_par)
        if hasattr(self,  "_bestmodel_fname"):
            log.info("Loading back the best model from epoch {} file {}".format(self._best_epoch,
                                                            self._bestmodel_fname))
            self._clf = load_model(C.OUTPUT_DIR + "/" + self._bestmodel_fname)
        else:
            self._clf = model
        return self

class GRU_Classifier(KerasClassifier):
    def __init__(self, embedding_matrix, feature_params, model_params, fit_params):
        super().__init__(embedding_matrix, feature_params, model_params, fit_params)
        self._dropout1 = model_params["dropout1"]
        self._gru1 = model_params["gru1"]
        self._optimizer = model_params["optimizer"]
        self._additional_dense = model_params.get("additional_dense")
        self._dense1 = model_params.get("dense1")
        self._dropout2 = model_params.get("dropout2")

    def _get_model(self):
        inp = Input(shape=(self._maxlen, ))
        x = self._get_embedding_layer()(inp)
        x = SpatialDropout1D(self._dropout1)(x)
        x = Bidirectional(GRU(self._gru1, return_sequences=True, go_backwards=False))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        if self._additional_dense:
            x = Dense(self._dense1, activation="relu")(x)
            x = Dropout(self._dropout2)(x)

        outp = Dense(6, activation="sigmoid")(conc)

        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='binary_crossentropy',
                    optimizer=self._optimizer,
                    metrics=['accuracy'],)
        return model

class CuDNN_GRU_Classifier(GRU_Classifier):
    def _get_model(self):
        inp = Input(shape=(self._maxlen, ))
        x = self._get_embedding_layer()(inp)
        x = SpatialDropout1D(self._dropout1)(x)
        x = Bidirectional(CuDNNGRU(self._gru1, return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        x = Dense(self._dense1, activation="relu")(x)
        x = Dropout(self._dropout2)(x)
        outp = Dense(6, activation="sigmoid")(conc)

        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='binary_crossentropy',
                    optimizer=self._optimizer,
                    metrics=['accuracy'],)
        return model

class CuDNN_LSTM_Classifier(KerasClassifier):
    '''
    Source: https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras
    '''
    def __init__(self, embedding_matrix, feature_params, model_params, fit_params):
        super().__init__(embedding_matrix, feature_params, model_params, fit_params)
        self._lstm1 = model_params["lstm1"]
        self._dropout1 = model_params["dropout1"]
        self._dense1 = model_params["dense1"]
        self._dropout2 = model_params["dropout2"]
        self._optimizer = model_params["optimizer"]
        self._bidirectional = model_params.get("bidirectional")

    def _get_model(self):
        inp = Input(shape=(self._maxlen, ))
        x = self._get_embedding_layer()(inp)
        x = Dropout(self._dropout1)(x)
        if self._bidirectional:
            x = Bidirectional(CuDNNLSTM(self._lstm1, return_sequences=True,name='lstm_layer'))(x)
        else:
            x = CuDNNLSTM(self._lstm1, return_sequences=True,name='lstm_layer')(x)
        x = GlobalMaxPool1D()(x)
        #x = Dropout(self._dropout1)(x)
        x = Dense(self._dense1, activation="relu")(x)
        x = Dropout(self._dropout2)(x)
        x = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy',
                        optimizer=self._optimizer,
                        metrics=['accuracy'])
        return model

class CuDNN_LSTM_WithConv_Spatial_Classifier(KerasClassifier):
    def __init__(self, embedding_matrix, feature_params, model_params, fit_params):
        super().__init__(embedding_matrix, feature_params, model_params, fit_params)
        self._lstm1 = model_params["lstm1"]
        self._dropout1 = model_params["dropout1"]
        self._dense1 = model_params["dense1"]
        self._dropout2 = model_params["dropout2"]
        self._optimizer = model_params["optimizer"]
        self._bidirectional = model_params.get("bidirectional")
        self._conv1 = model_params["conv1"]
        self._conv_kernel_size = model_params["conv_kernel_size"]

    def _get_model(self):
        inp = Input(shape=(self._maxlen, ))
        x = self._get_embedding_layer()(inp)
        x = SpatialDropout1D(self._dropout1)(x)
        if self._bidirectional:
            x = Bidirectional(CuDNNLSTM(self._lstm1, return_sequences=True,name='lstm_layer'))(x)
        else:
            x = CuDNNLSTM(self._lstm1, return_sequences=True,name='lstm_layer')(x)
        x = Conv1D(self._conv1, kernel_size=self._conv_kernel_size, padding='valid', kernel_initializer='glorot_uniform')(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])
        #x = Dropout(self._dropout1)(x)
        x = Dense(self._dense1, activation="relu")(x)
        x = Dropout(self._dropout2)(x)
        x = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy',
                        optimizer=self._optimizer,
                        metrics=['accuracy'])
        return model



class CNN_Classifier(KerasClassifier):
    '''
    taken from here : https://github.com/conversationai/unintended-ml-bias-analysis/blob/master/unintended_ml_bias/model_tool.py
    '''
    def __init__(self, embedding_matrix, feature_params, model_params, fit_params):
        super().__init__(embedding_matrix, feature_params, model_params, fit_params)
        self._cnn_filter_sizes = model_params["cnn_filter_sizes"]
        self._cnn_kernel_sizes = model_params["cnn_kernel_sizes"]
        self._cnn_pooling_sizes = model_params["cnn_pooling_sizes"]
        self._dropout1 = model_params["dropout1"]
        self._dense1 = model_params["dense1"]
        self._learning_rate  = model_params["learning_rate"]

    def _get_model(self):
        inp = Input( shape=(self._maxlen, ))

        x = self._get_embedding_layer()(inp)

        for filter_size, kernel_size, pool_size in zip(
            self._cnn_filter_sizes, self._cnn_kernel_sizes,
            self._cnn_pooling_sizes):
            x = self._build_conv_layer(x, filter_size, kernel_size, pool_size)

        x = Flatten()(x)
        x = Dropout(self._dropout1)(x)
        # TODO(nthain): Parametrize the number and size of fully connected layers
        x = Dense(self._dense1, activation='relu')(x)
        preds = Dense(6, activation='softmax')(x)

        rmsprop = RMSprop(lr=self._learning_rate)
        model = Model(inp, preds)
        model.compile(
            loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc'])
        return model

    def _build_conv_layer(self, input_tensor, filter_size, kernel_size, pool_size):
        output = Conv1D(
            filter_size, kernel_size, activation='relu', padding='same')(
                input_tensor)
        if pool_size:
            output = MaxPooling1D(pool_size, padding='same')(output)
        else:
        # TODO(nthain): This seems broken. Fix.
            output = GlobalMaxPooling1D()(output)
        return output







