import torch
import torch.nn as nn
from criterion.criterion import *

def build_criterion(opts, class_weights, printer = print):
    criterion = None
    if opts.loss_fn == 'ce':
        criterion = CrossEntropyLoss(opts, ignore_index=-1)
    elif opts.loss_fn == 'multitask':
        criterion = CrossEntropyLoss_multitask(opts)
    elif opts.loss_fn == "focal":
        criterion = FocalLoss()
    elif opts.loss_fn == "dsce": 
        criterion = DSCELoss(opts)
    elif opts.loss_fn == 'label_smoothing':
        criterion = LabelSmoothing()
    elif opts.loss_fn == 'ds':
        criterion = DSLoss()
    return criterion
        

