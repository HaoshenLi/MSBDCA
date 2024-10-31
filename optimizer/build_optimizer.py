import torch
from torch import optim

def build_optimizer(model, opts, printer=print):
    optimizer = None
    params = [p for p in model.parameters() if p.requires_grad]
    if opts.optim == 'sgd':
        optimizer = optim.SGD(params, lr=opts.lr, weight_decay=opts.weight_decay)
    elif opts.optim == 'adam':
        beta1 = 0.9 if opts.adam_beta1 is None else opts.adam_beta1
        beta2 = 0.999 if opts.adam_beta2 is None else opts.adam_beta2
        optimizer = optim.Adam(
            params,
            lr=opts.lr,
            betas=(beta1, beta2),
            weight_decay=opts.weight_decay,
            eps=1e-9)
    elif opts.optim == "adamw":
        beta1 = 0.9 if opts.adam_beta1 is None else opts.adam_beta1
        beta2 = 0.999 if opts.adam_beta2 is None else opts.adam_beta2
        optimizer = optim.AdamW(
            params,
            lr=opts.lr,
            betas=(beta1, beta2),
            weight_decay=opts.weight_decay,
            eps=1e-9)

    return optimizer


def update_optimizer(optimizer, lr_value):
    optimizer.param_groups[0]['lr'] = lr_value
    return optimizer


def read_lr_from_optimzier(optimizer):
    return optimizer.param_groups[0]['lr']


