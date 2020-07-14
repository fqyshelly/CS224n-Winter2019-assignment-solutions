#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, word_embed_size):
        super(Highway, self).__init__()
        self.W_project = nn.Linear(word_embed_size, word_embed_size, bias=True)
        self.W_gate = nn.Linear(word_embed_size, word_embed_size, bias=True)
        
    def forward(self, X_conv_out: torch.Tensor):
        X_proj = F.relu(self.W_project(X_conv_out))
        X_gate = torch.sigmoid(self.W_gate(X_conv_out))
        X_highway = X_gate * X_proj + (1-X_gate) * X_conv_out
        return X_highway

### END YOUR CODE 

