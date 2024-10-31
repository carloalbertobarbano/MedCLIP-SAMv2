"""
Author: Carlo Alberto Barbano (carlo.barbano@unito.it)
Date: 23/04/24
"""
import random
import os
import numpy as np
import torch

from clip.model import VisionTransformer

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def partial_forward(model: VisionTransformer, start_block, x):
    x = model.transformer.resblocks[start_block:](x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_post(x[:, 0, :])
    if model.proj is not None:
        x = x @ model.proj
    return x