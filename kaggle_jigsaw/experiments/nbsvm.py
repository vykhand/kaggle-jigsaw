import sys
import os

from kaggle_jigsaw import util as u
from kaggle_jigsaw.preprocessing.common import WordCharPreprocessor
from kaggle_jigsaw.modeling.linear import NbSvmClassifier
from kaggle_jigsaw.modeling.basic import BasicCrossValidator

import logging
log = logging.getLogger("jigsaw")

from sacred import Experiment
from sacred.observers import MongoObserver #, SqlObserver


#setting up sacred
ex_nb_svm = Experiment("NBSVM")
ex_nb_svm.observers.append(MongoObserver.create(url=os.environ["COSMOS_URL"],
                                db_name="kaggle-jigsaw",
                                collection="kaggle-jigsaw",
                                ssl="true" ))
#sacred_experiment.observers.append(SqlObserver.create(os.environ["AZURE_PGSQL"]))
ex_nb_svm.logger = log





@ex_nb_svm.automain
def run_nblr(exec_params, feature_params, model_params, fit_params,
    _seed, _run, _log ):

    #_log = log
    #_log.debug("Hola")

    prepr = WordCharPreprocessor(exec_params, feature_params)
    _run.info["cached_names"] = prepr.cached_file_names

    X_train, X_test, y_train = prepr.get_features()

    estimator = NbSvmClassifier(model_params)

    CV = BasicCrossValidator(X_train, X_test, y_train, estimator,
                                exec_params, fit_params, _run)

    result = CV.run()

    return result

