import os
import pickle
import re
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

import kaggle_jigsaw.util as u
import kaggle_jigsaw.constants as k

import logging
log = logging.getLogger("jigsaw")
from azureml.logging import get_azureml_logger
from collections import OrderedDict
from datetime import datetime
ml_log = get_azureml_logger()

t = u.get_timer()

class BasicCrossValidator:
    def __init__(self, X_train, X_test,
                    y_train, estimator, exec_params,
                    fit_params, _run):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self._model = estimator
        self._ex_par = exec_params
        self._fit_par = fit_params
        self.sample_sub = pd.read_csv(u.get_file("sample_submission.csv.zip"))
        os.makedirs(k.OUTPUT_DIR, exist_ok=True)
        try:
            self.num_folds = exec_params["num_folds"]
            self.debug_mode = exec_params["debug_mode"]
            self.skip_cv = exec_params["skip_cv"]
            self.retrain = exec_params["retrain"]
        except KeyError as e:
            log.error("Some mandatory keys are missing")
            raise e
        if self.debug_mode:
            self.X_test = self.X_test[:k.DEBUG_SAMPLES_TEST, :]
        self._aml_run_id = os.environ.get("AZUREML_RUN_ID", "local")
        if self._aml_run_id != "local":
            self._aml_run_id = re.findall(r"(\d+)",self._aml_run_id)[0]
        self._sacred_run_id = _run._id

    def get_folds(self):
        # to make it work fast on CPU
        if self.debug_mode:
            fold_fname = "debug_3folds.pkl"
        else:
            fold_fname = "{}_{}folds.pkl".format(k.FOLD_PREFIX, self.num_folds)

        self.fold_fname = fold_fname
        t.tic("Getting folds from file : {}".format(fold_fname))
        self.folds = pickle.load(open(u.get_file(fold_fname),"rb"))
        self.folds = OrderedDict(sorted(self.folds.items()))
        t.toc()

        return self.folds
    def run_full_train(self):
        t.tic("Training model on a full set")
        model = self._fit(self.X_train, self.y_train)
        t.toc()
        t.tic("Scoring test set")
        test_preds = model.predict_proba(self.X_test)
        t.toc()

        submission = pd.DataFrame(test_preds)

        log.info("Saving submission for model trained on full set")
        self.save_submission(submission, "no_cv", self._score)

    def _fit(self, X, y, xval=None, yval=None):
        ''' Fit method in case needs to be overriden'''
        return self._model.fit(X, y, xval, yval)


    def run_cv_iter(self, x, y, xval, yval, val_ind,  fold_name):

            t.tic("Fitting the model fold {}".format(fold_name))
            model = self._fit(x, y, xval, yval)
            t.toc()

            # save oof predictions to dict
            t.tic("Scoring OOF predictions fold {}".format(fold_name))
            preds = model.predict_proba(xval)
            self._oofs[val_ind,:] = preds
            t.toc()

            t.tic("Scoring test predictions fold {}".format(fold_name))
            self._test_preds[fold_name] = pd.DataFrame(model.predict_proba(self.X_test))
            t.toc()

            score = roc_auc_score(yval, preds, average=None)
            self._scores[fold_name] = score
            return model, score

    def run_cv(self):
        folds = self.get_folds()

        #roc aucss
        self._scores = {}
        # fold_name, pd.DataFrame with oofs
        self._oofs = np.zeros(self.y_train.shape)
        self._test_preds = {}

        t.get_msg("Starting cross validation loop, num_folds: {}".format(self.num_folds))
        for fold_name, inds in folds.items():
            train_ind, test_ind = inds
            Xtr, Xte = self.X_train[train_ind, :], self.X_train[test_ind,:]
            ytr, yte = self.y_train.iloc[train_ind, :], self.y_train.iloc[test_ind,:]

            model, score = self.run_cv_iter(Xtr, ytr, Xte, yte, test_ind, fold_name)

            log.info("Trained fold: {}, auc: {:.5f}, global time: {}".format(fold_name,
                                                                np.mean(score), t.get_global_time()))

        self._scores = pd.DataFrame(self._scores)
        mean_auc = np.mean(self._scores.mean())
        std_auc = np.std(self._scores.mean())

        t.get_msg("Finished training, mean_auc: {}, std_auc: {}".format(mean_auc,
                                                                    std_auc))

        ml_log.log("MeanAUC", mean_auc)
        ml_log.log("std_AUC", std_auc)

        self._oofs = pd.DataFrame(self._oofs, index = self.y_train.index,
                                        columns = self.y_train.columns)
        self._score = mean_auc
        self._score_std = std_auc

        self.save_outputs()

        # saving averaged submission
        sub  =  sum([df for df in self._test_preds.values()])/len(self._test_preds.keys())
        self.save_submission(sub, self.fold_fname, mean_auc)

        return mean_auc

    def run(self):
        "Main flow method"
        if self.skip_cv:
            t.tic("Skipping CV, training on full set")
            self._score = 0
            self.run_full_train()
            t.toc()
            return 0

        score = self.run_cv()

        if self.retrain:
            t.tic("Retraining on full dataset")
            self.run_full_train()
            t.toc()

        return score

    def save_outputs(self):
        "Saves files to outputs folder"
        ## save oofs and scores
        prefix = "{}_run_{}_{}_auc_{:.5f}".format(datetime.now().strftime("%Y%m%d_%H.%M"),
                                self._sacred_run_id, self._aml_run_id, self._score )

        with open(k.OUTPUT_DIR + "/" + prefix + "_OOFs.pkl", "wb") as f:
            pickle.dump(self._oofs, f)

        with open(k.OUTPUT_DIR + "/" + prefix + "_test_preds.pkl", "wb") as f:
            pickle.dump(self._test_preds, f)

        self._scores.to_csv(k.OUTPUT_DIR + "/" + prefix + "_scores.csv")

        self._file_prefix = prefix


    def save_submission(self, sub, fold_fname, score):
        '''
        submission is a pd.DataFrame
        '''
        sub_name = k.OUTPUT_DIR + "/{}_sub_run_{}_{}_{}_auc_{:.5f}.csv".format(datetime.now().strftime("%Y%m%d_%H.%M"),
                                                    self._sacred_run_id, self._aml_run_id, fold_fname, score )

        #l = self.test_preds[list(self.test_preds.keys())[0]].shape[0]

        sub = pd.concat([self.sample_sub.id, sub], axis = 1)
        sub.columns = self.sample_sub.columns

        self.submission = sub

        sub.to_csv(sub_name, index=False)

