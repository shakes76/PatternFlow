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

#%%
def diff(image, axis):

    ndim = image.ndim
    if ndim == 3:    
        if axis == 0:
            return image[1:,:,:] - image[:-1,:,:]
        elif axis == 1:
            return image[:,1:,:] - image[:,:-1,:]
        elif axis == 2:
            return image[:,:,1:] - image[:,:,:-1]
    elif ndim == 2: 
        if axis == 0:
            return image[1:,:] - image[:-1,:]
        elif axis == 1:
            return image[:,1:] - image[:,:-1]
#%%                    ]
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
            for ax in range(ndimt):
                slices_dt[ax] = slice(1, None)
                slices_pt[ax+1] = slice(0, -1)
                slices_pt[0] = ax
                dt[tuple(slices_dt)] += pt[tuple(slices_pt)]
                slices_dt[ax] = slice(None)
                slices_pt[ax+1] = slice(None)
            outt = ttt + dt
        else:
            outt = ttt
        Et = (dt ** 2).sum()
        
        slices_gt = [slice(None), ] * (ndimt + 1)
        for ax in range(ndimt):
            slices_gt[ax+1] = slice(0, -1)
            slices_gt[0] = ax
            gt[tuple(slices_gt)] = diff(outt, ax)
            slices_gt[ax+1] = slice(None)
        normt = torch.sqrt((gt ** 2).sum(axis=0)).unsqueeze(0)
        Et += weight * normt.sum()
        tau = 1. / (2.*ndimt)
        normt *= tau / weight
        normt += 1.
        pt -= tau * gt
        pt /= normt
        Et /= float(ttt.view(-1).shape[0])
        if i == 0:
            E_init = Et
            E_previous = Et
        else:
            if torch.abs(E_previous - Et) < eps * E_init:
                break
            else:
                E_previous = Et
        i += 1
        return outt
#%%
def denoise_tv_chambolle_torch(ttt, weight=0.1, eps=2.e-4, n_iter_max=200,
                         multichannel=False):
    
    imageType = ttt.dtype
    if imageType is not torch.float32:
        ttt = torch.FloatTensor(ttt)
    if multichannel:
        outt = torch.zeros_like(ttt)
        for c in range(ttt.shape[-1]):
            outt[...,c] = _denoise_tv_chambolle_nd_torch(ttt[..., c], weight, eps, n_iter_max)
    else:
        outt = _denoise_tv_chambolle_nd_torch(ttt, weight, eps, n_iter_max)
    
    return outt