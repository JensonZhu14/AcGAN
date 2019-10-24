import sys, time, json
import numpy as np
import torch
import random
import torch.nn.functional as F

def get_age_label(name, dataset):
    if 'fgnet' in dataset:
        age = int(name[4:6])
    elif 'morph' in dataset:
        str_M = name.split('M')
        str_F = name.split('F')
        if len(str_M) == 2:
            age = int(str_M[1][0:2])
        elif len(str_F) == 2:
            age = int(str_F[1][0:2])
    elif 'UTKFace' in dataset:
        files = name.split('_')
        age = int(files[0])
    elif 'CACD' in dataset:
        files = name.split('_')
        age = int(files[0])
    else:
        raise ValueError("Dataset %s is not exist." % dataset)

    return age

class Timer(object):
    """Timer class."""
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self.start = time.time()

    def stop(self):
        self.end = time.time()
        self.interval = self.end - self.start

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()


def load_params(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data


def print_params(model_name, params, datetime, dataset=None):
    print('----------------------------[%s]---------------------------' % model_name)
    if dataset is not None:
        print("Dataset: %s" % dataset)
        print("Running time: %s" % datetime)
    for i, (key, value) in enumerate(params.items()):
        print("%s: %s" % (key, str(value)))
    print('-------------------------------------------------------------')


def age_to_group(age, age_group=5):
    if age_group == 5:
        if age <= 20:
            label = 0
        elif 21 <= age <= 30:
            label = 1
        elif 31 <= age <= 40:
            label = 2
        elif 41 <= age <= 50:
            label = 3
        elif age > 50:
            label = 4
    elif age_group == 4:
        if age <= 30:
            label = 0
        elif 31 <= age <= 40:
            label = 1
        elif 41 <= age <= 50:
            label = 2
        elif age > 50:
            label = 3
    return label

'''
生成group->one hot，
'''
def desired_group_to_one_hot(group, age_group=5):
    group = group.reshape((-1, 1))
    bs = group.shape[0]
    one_hot_label = torch.zeros(bs, age_group)
    for i in range(bs):
        label = group[i, 0]
        label = random.choice([x for x in range(age_group) if x != label])
        one_hot_label[i, label] = 1
    return one_hot_label

def group_to_one_hot(group, age_group=5):
    group = group.reshape((-1, 1))
    bs = group.shape[0]
    one_hot_label = torch.zeros(bs, age_group)
    for i in range(bs):
        label = group[i, 0]
        one_hot_label[i, label] = 1
    return one_hot_label

def desired_group_to_binary(group, age_group=5):
    group = group.reshape((-1, 1))
    bs = group.shape[0]
    binary_label = torch.zeros(bs, age_group - 1)
    for i in range(bs):
        label = group[i, 0]
        label = random.choice([x for x in range(age_group) if x != label])
        binary_label[i, :label] = 1
    return binary_label

def group_to_binary(group, age_group=5):
    group = group.reshape((-1, 1))
    bs = group.shape[0]
    binary_label = torch.zeros(bs, age_group - 1)
    for i in range(bs):
        label = group[i, 0]
        binary_label[i, :label] = 1
    return binary_label

def age_to_one_hot(age, age_group=5):
    bs = age.shape[0]
    one_hot_label = torch.zeros(bs, age_group)
    for i in range(bs):
        label = age_to_group(age[i, 0], age_group)
        one_hot_label[i, label] = 1 
    return one_hot_label


def age_to_binary(age, age_group=5):
    bs = age.shape[0]
    binary_label = torch.zeros(bs, age_group - 1)
    for i in range(bs):
        label = age_to_group(age[i, 0], age_group)
        binary_label[i, :label] = 1
    return binary_label

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # m.weight.data.normal_(0, math.sqrt(2. / n))
        m.weight.data.normal_(0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.zero_()
