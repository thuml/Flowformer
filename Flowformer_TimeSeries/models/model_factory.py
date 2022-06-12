from typing import Optional, Any
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.Flowformer import FlowformerClassiregressor


def model_factory(config, data):
    task = config['task']
    feat_dim = data.feature_df.shape[1]  # dimensionality of data features
    # data windowing is used when samples don't have a predefined length or the length is too long
    max_seq_len = config['data_window_len'] if config['data_window_len'] is not None else config['max_seq_len']
    if max_seq_len is None:
        try:
            max_seq_len = data.max_seq_len
        except AttributeError as x:
            print(
                "Data class does not define a maximum sequence length, so it must be defined with the script argument `max_seq_len`")
            raise x

    if (task == "classification") or (task == "regression"):
        num_labels = len(data.class_names) if task == "classification" else data.labels_df.shape[1]
        if config['model'] == 'flowformer':
            return FlowformerClassiregressor(feat_dim, max_seq_len, config['d_model'],
                                             config['num_heads'],
                                             config['num_layers'], config['dim_feedforward'],
                                             num_classes=num_labels,
                                             dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                             activation=config['activation'],
                                             norm=config['normalization_layer'], freeze=config['freeze'])
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))
