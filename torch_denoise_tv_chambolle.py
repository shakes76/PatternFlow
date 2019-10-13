#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 14:30:46 2019

@author: s4503302
"""

import torch
import os
import matplotlib.pyplot as plt
import matplotlib.image as img

def _denoise_tv_chambolle_nd_torch(ttt, weight=0.1, eps=2.e-4, n_iter_max=200):
    
     ndimt = ttt.ndim
     pt = torch.zeros((ttt.ndim, ) + ttt.shape, dtype=ttt.dtype)
     gt = torch.zeros_like(pt)
     dt = torch.zeros_like(ttt)
     i = 0
     while i < n_iter_max:
        if i > 0:
            dt = -pt.sum(0)
            slices_dt = [slice(None), ] * ndimt
            slices_pt = [slice(None), ] * (ndimt + 1)