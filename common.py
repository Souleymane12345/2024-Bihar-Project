import os 


import yaml

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
#print(ROOT_DIR)
CONFIG_PATH = os.path.join(ROOT_DIR, 'config/config.yaml')

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

CURRENT_PATH = config['paths']['current_path']
MODEL_PATH = config['paths']['model_path']
LOG_PATH = config['paths']['logs']

PRED_API_PATH = config['paths']['predict']

DB = config['ml']['db']
DB_PRED = config['ml']['db_api']


