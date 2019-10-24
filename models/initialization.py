# -*- coding: utf-8 -*-
from copy import deepcopy
import os
import numpy as np
from .AcGANs import AcGANsModel

def initialize_model(config, datetime_str):
    print("Initializing model...")
    model_name = config['dataset']['model_name']
    model_config = config[model_name]
    model_param = deepcopy(model_config)
    model_param['save_time'] = datetime_str
    for k in config['dataset']:
        model_param[k] = config['dataset'][k]
    if model_name == "AcGANs":
        model = AcGANsModel(**model_param)
    else:
        raise NotImplementedError
    print("Model initialization complete.")
    return model


def initialization(config, datetime_str):
    os.environ["CUDA_VISIBLE_DEVICES"] = config["dataset"]["CUDA_VISIBLE_DEVICES"]
    return initialize_model(config, datetime_str)
