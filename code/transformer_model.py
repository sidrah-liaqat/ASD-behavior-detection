from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import optim, nn
import torch
import torch.nn.functional as F
import scipy.ndimage.filters as filter
import math
import clf

def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false
    In place operation
    :param tns:
    :return:
    """
    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval

class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()
        self.k = k
        self.t = seq_length
        self.num_tokens = num_tokens
        #self.token_emb = nn.Embedding(num_tokens, k)
        # here words are replaced by landmarks,
        # so replacing embedding with linear layer
        self.headpose = nn.Linear(6, 6)
        self.landmarks = nn.Linear(137, 30)
        self.eyelandmarks = nn.Linear(113, 30)
        self.au = nn.Linear(35,10)
        # k = 30+10=40
        self.pos_emb = nn.Embedding(seq_length, k)#k

        # The sequence of transformer blocks that does all the
        # heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        # Maps the final output sequence to class logits
        self.toprobs = nn.Linear(k, num_classes)

    def forward(self, dev, v, x, y, z):
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
        """
        # v.shape = (b, frame, 6) eyegaze also added along with headpose
        # x.shape = (b, frame, 137)
        # y.shape = (b, frame, 113)
        # z.shape = (b, frame, 35)

        tokens0 = self.headpose(v)
        tokens1 = self.landmarks(x)
        tokens2 = self.eyelandmarks(y)
        tokens3 = self.au(z)

        tokens = torch.cat([tokens0, tokens1, tokens2, tokens3], dim=2)
        #b, t, k = tokens.size()
        b=tokens.shape[0]

        # generate position embeddings
        positions = torch.arange(self.t)
        #TODO: Correct later so to(dev) can be removed
        #self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        positions = positions.to(dev)
        positions = self.pos_emb(positions)[None, :, :].expand(b, self.t, self.k)

        if(tokens.shape != positions.shape):
            print("pause")
        outp = tokens + positions #?? mathematical addition??
        outp = self.tblocks(outp)

        # Average-pool over the t dimension and project to class
        # probabilities
        #outp = self.toprobs(outp.mean(dim=1))
        outp = self.toprobs(outp)
        #return F.log_softmax(outp, dim=1)
        return outp

class SelfAttention(nn.Module):
    def __init__(self, k, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.k = k
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues = nn.Linear(k, k * heads, bias=False)

        self.unifyheads = nn.Linear(heads * k, k)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.k, f'Input embedding dim ({e}) should match layer embedding dim ({self.k})'

        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / math.sqrt(e) # dot contains b*h  t-by-t matrices with raw self-attention logits

        assert dot.size() == (b*h, t, t), f'Matrix has size {dot.size()}, expected {(b*h, t, t)}.'

        if self.mask: # mask out the lower half of the dot matrix,including the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2) # dot now has row-wise self-attention probabilities

        #assert not util.contains_nan(dot[:, 1:, :]) # only the forst row may contain nan
        assert not contains_nan(dot[:, 1:, :])  # only the forst row may contain nan

        if self.mask == 'first':
            dot = dot.clone()
            dot[:, :1, :] = 0.0
            # - The first row of the first attention matrix is entirely masked out, so the softmax operation results
            #   in a division by zero. We set this row to zero by hand to get rid of the NaNs

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        return x

