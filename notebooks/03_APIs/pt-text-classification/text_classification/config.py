import os
import sys
sys.path.append(".")
import logging
import logging.config

from text_classification import utils

APP_DIR = os.path.dirname(__file__)  # app root
app_path = APP_DIR.split("/")
# BASE_DIR is set dynamically based on the location of this file versus os.getcwd() as the model dependencies may
# not be served from the current working directory.
# There's probably a more Pythonic-way to determine BASE_DIR vs. string manipulation.
BASE_DIR = "/".join(app_path[:(len(app_path)-1)])

LOGS_DIR = os.path.join(BASE_DIR, 'logs')
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'embeddings')
EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments')

# Create dirs
utils.create_dirs(LOGS_DIR)
utils.create_dirs(EMBEDDINGS_DIR)
utils.create_dirs(EXPERIMENTS_DIR)

# Loggers
# Some logging issues ... when enabled, I see this error:
# FileNotFoundError: [Errno 2] No such file or directory: '/Users/dlite/projects/play/lessons/notebooks/03_APIs/pt-text-classification/build/logs/errors.log'
# log_config = utils.load_json(
#     filepath=os.path.join(BASE_DIR, 'logging.json'))
# logging.config.dictConfig(log_config)
# logger = logging.getLogger('logger')
