from azure.storage.blob import BlockBlobService
from azure.mgmt.compute import ComputeManagementClient
from azure.common.credentials import ServicePrincipalCredentials
from scipy.sparse import csr_matrix
import numpy as np
import hashlib
import json
import os
import sys
import psutil
import logging
from . import constants as C
import pandas as pd
import yaml
from time import time, sleep
from collections import deque

import logging
log = logging.getLogger("jigsaw")
#log = logging.getLogger()

from . import constants as C

from azureml.logging import get_azureml_logger
import logging

from sacred.observers import MongoObserver, TelegramObserver
# to set local env if not running on AML


def set_env_from_amlconfig():
    if "AZUREML_NATIVE_SHARE_DIRECTORY" not in os.environ:
        base_path = os.path.dirname(__file__)
        yaml_conf = yaml.load(
            open(os.path.join(base_path, "..", "aml_config", "local.compute"), "r"))
        os.environ["AZUREML_NATIVE_SHARE_DIRECTORY"] = yaml_conf["nativeSharedDirectory"] \
            + "/amlexp/amlexpWorkspace/kaggle-jigsaw"
        runconfig_env_vars = yaml.load(open(os.path.join(base_path, "..", "aml_config", "local.runconfig"),
                                            "r"))["EnvironmentVariables"]
        os.environ.update(runconfig_env_vars)


set_env_from_amlconfig()

def get_mongo_observer():
    return MongoObserver.create(url=os.environ["COSMOS_URL"],
                                db_name="kaggle-jigsaw",
                                collection="kaggle-jigsaw",
                                ssl="true" )

def get_telegram_observer():
    base_path = os.path.dirname(__file__)
    conf_file  = os.path.join(base_path, "..", "aml_config", "telegram_conf.json")
    obs = TelegramObserver.from_config(conf_file)
    # copied that from the source file and added RUN id
    obs.started_text = "♻ *RUN {_id} {experiment[name]}* " \
                    "started at _{start_time}_ " \
                    "on host `{host_info[hostname]}`"
    obs.completed_text = "✅ *RUN {_id} {experiment[name]}* " \
                            "completed after _{elapsed_time}_ " \
                            "with result=`{result}`"
    obs.interrupted_text = "⚠ *RUN {_id} {experiment[name]}* " \
                            "interrupted after _{elapsed_time}_"
    obs.failed_text = "❌ *RUN {_id} {experiment[name]}* failed after " \
                        "_{elapsed_time}_ with `{error}`\n\n" #\
                        #"Backtrace:\n```{backtrace}```"
    return obs

# TODO: bad practice, refactor
datadir = os.environ["AZUREML_NATIVE_SHARE_DIRECTORY"]
container_name = os.environ["AZURE_CONTAINER_NAME"]


class TicTocTimer:
    def __init__(self):
        self.start_time = time()
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)
        self.start_mem = self.get_mem()
        self.timings = deque()
        log.debug("Global timer set {}".format(self.start_time))

    def get_mem(self):
        return self.process.memory_info().rss

    def get_msg(self, msg):
        return "{}, global time: {:.2f}, mem use:{:.2f}GB".format(msg,
                                                                  time() - self.start_time,
                                                                  self.get_mem()/2**30)
    def get_diff_msg(self, msg, t, mem):
        return msg + \
        " time:{:.2f} global_time:{:.2f} step_mem:{:.2f}MB mem_use:{:.2f}GB".format(time()-t,
        time() - self.start_time,
        (self.get_mem()-mem)/2**20,
        self.get_mem()/2**30)

    def tic(self, msg):
        self.timings.appendleft((msg, time(), self.get_mem()))
        log.info("Started:" + self.get_msg(msg))

    def toc(self, vals = None):
        msg, t, mem = self.timings.popleft()
        if vals is not None:
            msg=msg.format(vals)
        log.info("Finished: " + self.get_diff_msg(msg, t, mem))

    def get_global_time(self):
        return time() - self.start_time


timer = TicTocTimer()


def get_timer():
    return timer


def get_blob_service():
    blob = BlockBlobService(
        os.environ["AZURE_STORAGE_ACCOUNT"], os.environ["AZURE_STORAGE_KEY"])
    return blob


blob = get_blob_service()


def get_file_from_blob(file_name):
    file_path = os.path.join(datadir, file_name)
    blob.get_blob_to_path(
        os.environ["AZURE_CONTAINER_NAME"], os.path.basename(file_path), file_path)


def put_file_to_blob(file_name):
    file_path = os.path.join(datadir, file_name)
    blob.create_blob_from_path(
        os.environ["AZURE_CONTAINER_NAME"], os.path.basename(file_path), file_path)


def get_files(required_files):
    for f in required_files:
        get_file(f)


def is_available(file_name):
    files = [n.name for n in blob.list_blobs(
        container_name) if n.name == file_name]
    return len(files) > 0


def get_file(file_name):
    '''
    check if the file is in data dir directory
    '''
    fpath = os.path.join(datadir, file_name)

    if not os.path.exists(fpath):
        if is_available(file_name):
            get_file_from_blob(file_name)
        else:
            raise FileNotFoundError(
                "File {} not found in blob store".format(file_name))

    return fpath


def shutdown_vm():
    #sleep(C.SHUTDOWN_TIMEUT)
    try:
        vm_name = os.environ["AZURE_VM_NAME"]
        res_group = os.environ["AZURE_RESOURCE_GROUP"]
        client_id = os.environ['AZURE_CLIENT_ID']
        secret = os.environ['AZURE_CLIENT_SECRET']
        tenant = os.environ['AZURE_TENANT_ID']
        subscription_id = os.environ['AZURE_SUBSCRIPTION_ID']

        credentials = ServicePrincipalCredentials(client_id=client_id,
                                                  secret=secret,
                                                  tenant=tenant)

        compute_client = ComputeManagementClient(credentials, subscription_id)

        compute_client.virtual_machines.deallocate(res_group, vm_name )
        log.info("Requested to deallocate VM {} group {}".format(
            res_group, vm_name))
    except KeyError as e:
        log.warning("Can't read Azure config to shutdown VM")
        log.warning(str(e))
    except Exception as e:
        log.warning("Unable to shutdown VM")
        log.warning(str(e))


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def set_console_logger(log):
    log.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(name)s:[%(asctime)s] {%(module)s.%(funcName)s:%(lineno)d %(levelname)s} - %(message)s')
    #'%m-%d %H:%M:%S'
    handl = logging.StreamHandler(stream=sys.stdout)
    handl.setFormatter(formatter)
    log.addHandler(handl)
    return log


def dict_to_md5(d):
    '''
    converts dict to md5 hash in a consistent manner
    '''
    s = json.dumps(d, sort_keys=True)
    return hashlib.md5(s.encode('utf-8')).hexdigest()
