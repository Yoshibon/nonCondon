import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from collections import OrderedDict


class SkipConnection(nn.Module):
    def __init__(self, dim_model, dropout=0.0):
        super(SkipConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()
        self.linear = nn.Linear(dim_model, dim_model)
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        x = self.dropout(x)
        hidden = x + self.relu(self.linear(x))
        output = self.norm(hidden)
        return output


class DeepLearningBaseline(nn.Module):
    def __init__(self, dim_in, dim_hidden, layer_num, dropout=0.0, dim_reg=16):
        super(DeepLearningBaseline, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.layer_num = layer_num
        self.dropout = dropout
        self.dim_reg = dim_reg

        # Input Embedding
        self.embedding = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.LeakyReLU()
        self.norm = nn.LayerNorm(dim_hidden)
        self.sigmoid = nn.Sigmoid()

        # Feature Model
        self.feature_model = nn.Sequential(OrderedDict([('feature_hidden_1', SkipConnection(dim_hidden, dropout))]))
        for i in range(layer_num - 1):
            self.feature_model.add_module('feature_hidden_{}'.format(i + 2), SkipConnection(dim_hidden, dropout))

        # Classifier
        self.reg = nn.Linear(self.dim_hidden, self.dim_reg)
        self.output = nn.Linear(self.dim_reg, 1)

    def forward(self, x):
        initial_embedding = self.norm(self.relu(self.embedding(x)))
        # initial_embedding = self.embedding(x)
        feature = self.feature_model(initial_embedding)
        hidden = self.relu(self.reg(feature))
        output = self.output(hidden)
        # output = torch.exp(output)
        output = self.sigmoid(output)
        return output


class MagnitudeModel(nn.Module):
    def __init__(self, input_dim, dim_hidden, layer_num, dropout, dim_reg):
        super(MagnitudeModel, self).__init__()
        self.model_C_CC_head = DeepLearningBaseline(input_dim[0], dim_hidden, layer_num, dropout, dim_reg)
        self.model_C_CO_head = DeepLearningBaseline(input_dim[1], dim_hidden, layer_num, dropout, dim_reg)
        self.model_G_CO_head = DeepLearningBaseline(input_dim[2], dim_hidden, layer_num, dropout, dim_reg)
        self.model_C_CC_body = DeepLearningBaseline(input_dim[0], dim_hidden, layer_num, dropout, dim_reg)
        self.model_C_CO_body = DeepLearningBaseline(input_dim[1], dim_hidden, layer_num, dropout, dim_reg)
        self.model_G_CO_body = DeepLearningBaseline(input_dim[2], dim_hidden, layer_num, dropout, dim_reg)

    def forward(self, x_ccc_head, x_ccc_body, x_cco_head, x_cco_body, x_gco_head, x_gco_body):
        magnitude_C_CC_head = self.model_C_CC_head(x_ccc_head)
        magnitude_C_CO_head = self.model_C_CO_head(x_cco_head)
        magnitude_G_CO_head = self.model_G_CO_head(x_gco_head)
        magnitude_C_CC_body = self.model_C_CC_body(x_ccc_body)
        magnitude_C_CO_body = self.model_C_CO_body(x_cco_body)
        magnitude_G_CO_body = self.model_G_CO_body(x_gco_body)

        return magnitude_C_CC_head, magnitude_C_CC_body, magnitude_C_CO_head, magnitude_C_CO_body, \
               magnitude_G_CO_head, magnitude_G_CO_body,


