#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import os


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio=2, fused=True):
        super(ConvBlock, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv1d(inp, hidden_dim, 9, 1, padding=4, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=False),
            nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm1d(oup),
        )

    def forward(self, x):
        return x + self.conv(x)


from torch import nn

from models.transformer import EncoderLayer


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        # pos = torch.arange(0, max_len)
        pos = torch.arange(0, max_len).cuda()
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float().cuda()
        # _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):

        return self.encoding[:x.shape[1], :].cuda()

class Moss(nn.Module):
    def __init__(self, kernel_num, topk):
        """
        Parameters
        ----------
        """
        super(Moss, self).__init__()
        self.kernel_num = kernel_num
        self.topk = topk
        self.detect_word = nn.Sequential(
            nn.Conv1d(4, self.kernel_num, stride=3, kernel_size=7, padding=2),
            nn.BatchNorm1d(self.kernel_num),
        )
        self.word_embedding = nn.Embedding(self.kernel_num, 256)
        self.cls_embedding = nn.Embedding(1, 256)
        self.positional_encoding = PositionalEncoding(256, 168)

        self.layers = nn.ModuleList([EncoderLayer(d_model=256,
                                                  ffn_hidden=256,
                                                  n_head=8,
                                                  drop_prob=0.2)
                                     for _ in range(6)])

        self.final = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        """Forward propagation of a batch.
        """
        x = F.one_hot(x, num_classes=4).transpose(1, 2).float()

        word_detect_conv = self.detect_word(x)
        word_detect_pro = F.softmax(word_detect_conv, dim=2)
        # print(word_detect_pro.shape)
        word_sort_values, word_sort_indices = torch.topk(word_detect_pro, self.topk, dim=-2)
        word_sort_values = word_sort_values.transpose(-1, -2)
        word_sort_indices = word_sort_indices.transpose(-1, -2)
        # print(word_sort_values.shape)
        # print(word_sort_indices.shape)
        all_embeddings = self.word_embedding(word_sort_indices)
        all_embeddings_probabilities = F.softmax(word_sort_values, dim=-1).unsqueeze(-1)
        word_embedding = torch.sum(all_embeddings * all_embeddings_probabilities, dim=2)

        sequence_embedding = self.cls_embedding(torch.zeros((all_embeddings.shape[0], 1),
                                                            dtype=torch.int, device=x.device))
        word_embedding = torch.concat([sequence_embedding, word_embedding], dim=1)

        position_encoding = self.positional_encoding(word_embedding)

        x = word_embedding + position_encoding
        atts = []
        for layer in self.layers:
            x, att = layer(x, None)
            atts.append(att)

        x_final = x[:, 0]
        out = self.final(x_final)

        return out, x_final       # plot umap
