from models.initialization import initialization
import sys, os, torch
from models.utils import ops
from datetime import datetime
import argparse
import yaml

# load configure
if len(sys.argv) <= 1:
    raise Exception("configure file must be specified!")
# config = ops.load_params(json_file=sys.argv[1])
with open(sys.argv[1]) as f:
    config = yaml.load(f)

model_name = config['dataset']['model_name']
dataset = config['dataset']['dataset_name']
if config[model_name]['restore_epoch'] != 0:
    datetime_str = config[model_name]['resume_time']
else:
    datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# save log 
log_path = './results/' + model_name + '/' + dataset + '/train_logs/'
os.makedirs(log_path, exist_ok=True)
logger = ops.Tee(log_path + datetime_str + '.txt', 'a')

# print config info
ops.print_params(model_name, config[model_name], datetime_str, dataset=dataset)

model = initialization(config, datetime_str)
print("Training START.")
model.fit()
print("Training COMPLETE.")
