import sys
import os

from kaggle_jigsaw import util as u
from kaggle_jigsaw.preprocessing.common import WordCharPreprocessor
from kaggle_jigsaw.modeling.basic import BasicCrossValidator
from sklearn.linear_model import LogisticRegression
import logging
log = logging.getLogger("jigsaw")

from sacred import Experiment
from sacred.observers import MongoObserver #, SqlObserver


#setting up sacred
ex_logreg = Experiment("LogReg")
ex_logreg.observers.append(MongoObserver.create(url=os.environ["COSMOS_URL"],
                                db_name="kaggle-jigsaw",
                                collection="kaggle-jigsaw",
                                ssl="true" ))
#sacred_experiment.observers.append(SqlObserver.create(os.environ["AZURE_PGSQL"]))
ex_logreg.logger = log


@ex_logreg.config
def config():
    feature_params = {"wordvec_params":
                        {"ngram_range":(1,3),
                        "strip_accents":"unicode",
                        "min_df": 5,
                     #   "max_df": .9,
                        "use_idf":    1,
                        "smooth_idf": 1,
                        "token_pattern": r"\w{1,}",
                        "sublinear_tf":   True,
                        "max_features":250000},
                      "charvec_params":
                            {"sublinear_tf":True,
                            "strip_accents":"unicode",
                            "min_df" : 3,
                            "ngram_range":(1, 6),
                            "max_features":250000},
                            }


    model_params ={"C":1.0,
                    "solver":"sag",
                    "multi_class":"ovr" }


    exec_params = { "num_folds": 5,
                    "cached_prefix" : "simple_logreg",
                    "skip_cv": False,
                    "retrain": False,
                    "debug_mode": False,
                    "disable_cache": False }
    fit_params = {}


@ex_logreg.automain
def run_logreg(exec_params, feature_params, model_params, fit_params,
    _seed, _run, _log ):

    #_log = log
    #_log.debug("Hola")

    prepr = WordCharPreprocessor(exec_params, feature_params)
    _run.info["cached_names"] = prepr.cached_file_names

    X_train, X_test, y_train = prepr.get_features()

    estimator = LogisticRegression(**model_params)

    CV = BasicCrossValidator(X_train, X_test, y_train, estimator,
                                exec_params, fit_params,  _run)

    result = CV.run()

    return result

