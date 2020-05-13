#%%
from typing import *

import torch
#%%
def get_ffn_net(
    feature_count,
    dropout = 0.1) -> torch.nn.Sequential:
    units = 64
    n_classes = 3
    hidden_layers_amount = 8
    model_args = [
        torch.nn.Linear(feature_count, units),
        torch.nn.SELU(),
        torch.nn.AlphaDropout(dropout),
    ]
    for layer_index in range(hidden_layers_amount - 1):
        model_args.extend([
            torch.nn.Linear(units, units),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(dropout)
        ])
    model_args.append(torch.nn.Linear(units, n_classes))
    return torch.nn.Sequential(*model_args)

def get_conv_net() -> torch.nn.Sequential:
    model_args = [
    ]
    return torch.nn.Sequential(*model_args)