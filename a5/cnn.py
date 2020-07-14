#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, char_embed_size, word_embed_size, kernel_size=5):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(char_embed_size, word_embed_size, kernel_size = kernel_size, stride=1)
    
    def forward(self, X_reshapped: torch.Tensor): #(w_batch, e_char, m_w)
        X_conv = self.conv(X_reshapped)
        X_conv_out,_ = torch.max(F.relu(X_conv), dim=-1)
        return X_conv_out #

### END YOUR CODE

#test
#if __name__ == '__main__':
 #   input = torch.randn(4, 3, 4)
 #   model = CNN(3,4,2)
 #   x_con = model(input)
 #   print(x_con.shape)
