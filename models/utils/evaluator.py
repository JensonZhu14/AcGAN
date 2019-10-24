# -*- coding:utf-8 -*-
import torch
import numpy as np
from torch.nn.modules.loss import _Loss


class MSELoss(_Loss):
    def forward(self, pred, target):
        loss = torch.sum((pred - target)**2) / target.size(0)
        return loss

class L1Loss(_Loss):
    def forward(self, pred, target):
        loss = torch.sum(torch.abs(pred - target)) / target.size(0)
        return loss

class OrderRegressionLoss(_Loss):
    def forward(self, pred, target):
        offset = 1e-10
        loss = - torch.sum(torch.log(pred + offset).mul(target) + torch.log(1 - pred + offset).mul(1 - target)) / pred.size(0)
        return loss

class CrossEntropyLoss(_Loss):
    def forward(self, pred, target):
        offset = 1e-10
        loss = - torch.sum(torch.log(pred + offset).mul(target)) / pred.size(0)
        return loss
