import sys

import os
import argparse

import logging
log = logging.getLogger("jigsaw")


import kaggle_jigsaw.util as u

import atexit
import yaml


if __name__ == "__main__":
    log = u.set_console_logger(log)

    # parameters
    # add experiment arguments
    parser = argparse.ArgumentParser()


    parser.add_argument('--model', choices=["GRU", "LSTM", "CNN", "LGB", "NBSVM", "LSTM_CONV",
                                            "LR", "RIDGE", "ET", "RF", "GBT", "KNN"
                                            ] ,
                                    help='Model type to run')
    parser.add_argument('--skip_cv', action="store_true", help='skip cv and train on the whole set')
    parser.add_argument('--retrain', action="store_true", help='train on the whole set')
    parser.add_argument('--shutdown_vm', action="store_true", help='shutdown VM after finishing the run')
    parser.add_argument('--num_folds', type=int, default=5,  help='number of folds for cross val')
    parser.add_argument('--conf',   help='number of folds for cross val')

    args = parser.parse_args()
    log.info("Running with arguments: {}".format(args))

    model = args.model
    if args.model is None:
        conf = yaml.load(open(args.conf,"r"))
        model = conf.get("exec_params",{}).get("model_type")
        log.debug("model_type = "+ model)
    if model is None:
        raise ValueError("Model type not configured neither via --model nor in the config")

    if model in ["GRU", "LSTM", "CNN", "LSTM_CONV"]:
        from kaggle_jigsaw.experiments.NN import NN_experiment
        experiment = NN_experiment
    if model in ["LR", "RIDGE", "ET", "RF", "GBT", "KNN"]:
        from kaggle_jigsaw.experiments.sklearn import sklearn_experiment
        experiment = sklearn_experiment
    elif model in ["LGB"]:
        from kaggle_jigsaw.experiments.lgb import lgb_experiment
        experiment = lgb_experiment
    elif model in ["NBSVM"]:
        from kaggle_jigsaw.experiments.nbsvm import ex_nb_svm
        experiment = ex_nb_svm
    else:
       raise ValueError("unknown model type")

    if args.conf is not None:
        experiment.add_config(args.conf)
    else:
        experiment.add_config("conf_template/{}.yaml".format(args.model))

    if args.shutdown_vm: atexit.register(u.shutdown_vm)

    experiment.run(config_updates={"exec_params":
                                    {"model_type": model,
                                     "skip_cv":args.skip_cv,
                                     "retrain": args.retrain,
                                     "num_folds":args.num_folds}} )
