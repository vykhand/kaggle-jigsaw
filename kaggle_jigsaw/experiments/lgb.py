import sys
import os

from kaggle_jigsaw import util as u
from kaggle_jigsaw.preprocessing.common import WordCharPreprocessor
from kaggle_jigsaw.preprocessing.trees import CntWordCharPreprocessor
from kaggle_jigsaw.modeling.trees import LGBClassiffier, LGBCrossValidator

import logging
log = logging.getLogger("jigsaw")

from sacred import Experiment


#setting up sacred
lgb_experiment = Experiment("LGB")
lgb_experiment.observers.append(u.get_mongo_observer())
lgb_experiment.observers.append(u.get_telegram_observer())
#sacred_experiment.observers.append(SqlObserver.create(os.environ["AZURE_PGSQL"]))
lgb_experiment.logger = log




@lgb_experiment.automain
def run_lgb(exec_params, feature_params, model_params, fit_params,
    _seed, _run, _log ):

    #_log = log
    #_log.debug("Hola")
    use_cnt = feature_params.get("use_cnt_features",False)
    if use_cnt:
        log.debug("Using count features")
        prepr = CntWordCharPreprocessor(exec_params, feature_params)
    else:
        log.debug("Not Using count features")
        prepr = WordCharPreprocessor(exec_params, feature_params)

    _run.info["cached_names"] = prepr.cached_file_names

    X_train, X_test, y_train = prepr.get_features()

    estimator = LGBClassiffier(model_params, fit_params)

    CV = LGBCrossValidator(X_train, X_test, y_train, estimator,
                                exec_params, fit_params, _run)

    result = CV.run()

    return result

