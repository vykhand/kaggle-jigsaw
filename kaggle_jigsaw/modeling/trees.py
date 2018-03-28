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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from scipy import sparse

import lightgbm as lgb
import numpy as np
from kaggle_jigsaw import constants as C
import kaggle_jigsaw.constants as k
import yaml

class LGBCrossValidator(BasicCrossValidator):
    '''
    overriden to allow for watching val score
    '''
    #def _fit(self, X, y, Xval=None, yval=None):
    #    if X is not None and y is not None:
    #        ret = self._model.fit(X, y, Xval, yval)
    #    else:
    #        ret = self._model.fit(X, y)
    #    return ret

    def _fit(self, X, y, xval=None, yval=None):
        # if running after CV
        mean_iters = None
        if (xval is None or yval is None) and \
            len(self._best_iters) > 0:
            mean_iters = pd.DataFrame(list(self._best_iters.values())).mean().apply(int).to_dict()
        return self._model.fit(X, y, xval, yval, mean_iters)
    def run_cv(self):
        self._best_iters = {}
        score = super().run_cv()
        return score

    def run_cv_iter(self,  x, y, xval, yval, val_ind,  fold_name):
        model, score = super().run_cv_iter(x, y, xval, yval, val_ind, fold_name)
        self._best_iters[fold_name] = model._best_iters
        return model, score
    def save_outputs(self):
        super().save_outputs()
        with open(k.OUTPUT_DIR + "/" + self._file_prefix + "_best_iters.yml", "w") as f:
            yaml.dump(self._best_iters, f, default_flow_style=False)




class LGBClassiffier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_params, fit_params):
        self._fit_par = fit_params
        self._lgb_params = model_params["lgb_params"]
        self._rounds_lookup = model_params["rounds_lookup"]
        self._early_stop_lookup = model_params["early_stop_lookup"]
        self._use_logreg_selector = model_params["use_logreg_selector"]
        self._selector_threshold  = model_params["selector_threshold"]

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_clf'])
        return self.predict_proba(x)

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, [ '_clf','_sel', '_best_iters'])
        preds = np.zeros((x.shape[0],len(self._clf.keys())))
        for k in self._clf.keys():
            if self._use_logreg_selector:
                l_x = self._sel[k].transform(x)
            else: l_x = x
            preds[:,k] = self._clf[k].predict(l_x,
                        num_iteration = self._best_iters.get(k, 0))
        return preds

    def fit(self, x, y, xval = None, yval =  None, mean_iters = None):
        # Check that X and y have correct shape
        self._clf = {}
        #x, y = check_X_y(x, y, accept_sparse=True, multi_output=True)

        assert(tuple(y.columns.values) ==  C.class_names)
        log.debug("Training on dataset shape:{}".format(x.shape))

        self._sel = {}
        self._best_iters = {}
        for i,c in enumerate(C.class_names):
            if self._use_logreg_selector:
                t.tic("Running feature selection with LogReg")
                l = LogisticRegression(solver = "sag")
                m = SelectFromModel(l, threshold = self._selector_threshold)
                self._sel[i] = m.fit(x, y[c])
                l_x = m.transform(x)
                if xval is not None:
                    l_xval = m.transform(xval)
                log.debug("selected features shape: {}".format(l_x.shape))
                t.toc()
            else:
                l_x = x
                l_xval = xval

            d_train = lgb.Dataset(l_x, label = y[c])
            val_set = [d_train]
            t.tic("Training model on class: " + c )
            if xval is not None and yval is not None:
                d_val = lgb.Dataset(l_xval, label = yval[c])
                val_set.append(d_val)


                model = lgb.train(params = self._lgb_params,
                                train_set = d_train,
                                num_boost_round = self._rounds_lookup[c],
                                early_stopping_rounds = self._early_stop_lookup.get(c),
                                valid_sets = val_set, **self._fit_par)
                self._clf[i] = model
                if hasattr(model, "best_iteration"):
                    self._best_iters[c] = model.best_iteration
                    log.debug("Best iteration for class {} is {}".format(c, model.best_iteration))
            else:
                rounds_lookup = mean_iters if mean_iters is not None \
                                           else self._rounds_lookup
                model = lgb.train(params = self._lgb_params,
                                    train_set = d_train,
                                    num_boost_round = rounds_lookup[c],
                                    valid_sets = val_set, **self._fit_par)
            self._clf[i] = model
            t.toc()
        return self