import sys

import os
import argparse

import logging
log = logging.getLogger("jigsaw")

import kaggle_jigsaw.util as u
import subprocess


if __name__ == "__main__":
    log = u.set_console_logger(log)

    # parameters
    # add experiment arguments
    parser = argparse.ArgumentParser()


    parser.add_argument('folder', help='folder to run experiments')
    parser.add_argument('compute',  help='compute target')
    #parser.add_argument('--dry',  action="store_true")
    parser.add_argument('args', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    log.info("Running with arguments: {}".format(args))

    for f in os.listdir(args.folder):
        cmd = "az ml experiment submit -c {} --wait run.py --conf {} {} ".format(args.compute,
                                                                                args.folder + "/" + f,
                                                                                 " ".join(args.args) )
        log.info("Running: " + cmd)
        os.system(cmd)

    cmd = "az vm deallocate -g {} -n {}".format(args.compute, args.compute)
    log.info("Running: " + cmd)
    os.system(cmd)
