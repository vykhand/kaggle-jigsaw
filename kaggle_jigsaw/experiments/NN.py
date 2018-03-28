import sys
import os

from kaggle_jigsaw import util as u
from kaggle_jigsaw.preprocessing.neural import KerasWithEmbeddings
from kaggle_jigsaw.preprocessing.embeddings import GloveEmbeddingMatrix, Word2VecEmbeddingsMatrix, FastTextEmbeddingsMatrix
from kaggle_jigsaw.modeling.basic import BasicCrossValidator
from kaggle_jigsaw.modeling.neural import KerasCrossValidator, CuDNN_GRU_Classifier
from kaggle_jigsaw.modeling.neural import CuDNN_LSTM_Classifier, CNN_Classifier
from kaggle_jigsaw.modeling.neural import CuDNN_LSTM_WithConv_Spatial_Classifier

import logging
log = logging.getLogger("jigsaw")

from sacred import Experiment
#, SqlObserver


#setting up sacred

NN_experiment = Experiment("NN_experiment")
NN_experiment.observers.append(u.get_mongo_observer())
NN_experiment.observers.append(u.get_telegram_observer())
#sacred_experiment.observers.append(SqlObserver.create(os.environ["AZURE_PGSQL"]))
NN_experiment.logger = log


@NN_experiment.automain
def run_NN_experiment(exec_params, feature_params, model_params, fit_params,
    _seed, _run, _log ):
    '''
    flow is defined by exec_params.model_type
    '''
    model_type = exec_params["model_type"]
    log.debug("model_type = " + model_type)
    #choosing model
    model_dict = {"GRU": CuDNN_GRU_Classifier,
                  "LSTM": CuDNN_LSTM_Classifier,
                  "LSTM_CONV": CuDNN_LSTM_WithConv_Spatial_Classifier,
                  "CNN": CNN_Classifier}
    assert(model_type in model_dict.keys())

    _run.experiment_info["name"] = exec_params["model_type"]

    embedding_type = feature_params.get("embedding_type", "glove")
    assert(embedding_type in ["glove", "fasttext", "word2vec"])

    #embedding type
    transf_dict = {"glove": GloveEmbeddingMatrix,
                    "fasttext": FastTextEmbeddingsMatrix,
                    "word2vec": Word2VecEmbeddingsMatrix}

    prepr = KerasWithEmbeddings(transf_dict[embedding_type], exec_params, feature_params)

    X_train, X_test, y_train, embed_matrix = prepr.get_features()



    estimator = model_dict[model_type](embed_matrix, feature_params,
                                 model_params, fit_params)

    CV = KerasCrossValidator(X_train, X_test, y_train, estimator,
                                exec_params, fit_params, _run)

    result = CV.run()

    return result

