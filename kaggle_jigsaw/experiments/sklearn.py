import sys
import os

from kaggle_jigsaw import util as u
from kaggle_jigsaw.preprocessing.common import WordCharPreprocessor
from kaggle_jigsaw.preprocessing.trees import CntWordCharPreprocessor
from kaggle_jigsaw.modeling.linear import BaseSklearnEstimator
from kaggle_jigsaw.modeling.basic import BasicCrossValidator

import logging
log = logging.getLogger("jigsaw")

from sacred import Experiment


#setting up sacred
sklearn_experiment = Experiment("SKlearn_Generic")
sklearn_experiment.observers.append(u.get_mongo_observer())
sklearn_experiment.observers.append(u.get_telegram_observer())
#sacred_experiment.observers.append(SqlObserver.create(os.environ["AZURE_PGSQL"]))
sklearn_experiment.logger = log

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


@sklearn_experiment.automain
def run_sklearn(exec_params, feature_params, model_params, fit_params,
    _seed, _run, _log ):

    model_type = exec_params["model_type"]
    log.debug("model_type = " + model_type)
    #choosing model
    model_dict = {"LR": LogisticRegression,
                  "RIDGE": RidgeClassifier,
                  "ET": ExtraTreesClassifier,
                  "RF": RandomForestClassifier,
                  "GBT": GradientBoostingClassifier,
                  "KNN": KNeighborsClassifier}
    assert(model_type in model_dict.keys())

    _run.experiment_info["name"] = exec_params["model_type"]

    use_cnt = feature_params.get("use_cnt_features",False)
    if use_cnt:
        log.debug("Using count features")
        prepr = CntWordCharPreprocessor(exec_params, feature_params)
    else:
        log.debug("Not Using count features")
        prepr = WordCharPreprocessor(exec_params, feature_params)

    _run.info["cached_names"] = prepr.cached_file_names

    X_train, X_test, y_train = prepr.get_features()

    estimator = BaseSklearnEstimator(model_dict[model_type], model_params)

    CV = BasicCrossValidator(X_train, X_test, y_train, estimator,
                                exec_params, fit_params, _run)

    result = CV.run()

    return result

