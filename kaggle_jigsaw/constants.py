import os

DEBUG_SAMPLES_TEST = 100
OUTPUT_DIR = "./outputs"
BADWORDS_FNAME = "full-list-of-bad-words-banned-by-google-txt-file_2013_11_26_04_53_31_867.txt"
#DATADIR = "C:/00.DATA/kaggle-jigsaw"
TRAIN_FILE = "train.csv.zip"
TEST_FILE = "test.csv.zip"
#AZURE_STORAGE_ACCOUNT = "expcompetitionsstorage"
#AZURE_STORAGE_KEY = "hbNjOsTtZmi85r1ff8JoywlmNhpY8Xw0rEYlTxnT2x/fwJFDbopBK/2eqsHIX2bjoOtBrEf/WKbvYtG3rxterA=="
#AZURE_CONTAINER_NAME  = "kaggle-jigsaw"
#FOLD_PREFIX = "packbit"
FOLD_PREFIX = "simple"
COMMENT = 'comment_text'
COMPETITION_NAME = "jigsaw-toxic-comment-classification-challenge"
class_names  = ('toxic',
                 'severe_toxic',
                 'obscene',
                 'threat',
                 'insult',
                 'identity_hate')
SHUTDOWN_TIMEOUT = 120

APOSTROPHES_FILE = "apostrophes.json"