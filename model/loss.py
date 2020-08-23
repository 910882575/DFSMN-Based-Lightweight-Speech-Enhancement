#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn.functional as F 


def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True) 
    data = data - mean
    return data

def pow_p_norm(signal):
    """Compute 2 Norm"""
    return torch.pow(torch.norm(signal, p=2, dim=-1, keepdim=True), 2)


def pow_norm(s1, s2):
    return torch.sum(s1 * s2, dim=-1, keepdim=True)


def si_snr(estimated, original,EPS=1e-8):
    # estimated = remove_dc(estimated)
    # original = remove_dc(original)
    target = pow_norm(estimated, original) * original / (pow_p_norm(original) + EPS)
    noise = estimated - target
    sdr = 10 * torch.log10(pow_p_norm(target) / (pow_p_norm(noise) + EPS) + EPS)
    return torch.mean(sdr)

def sd_snr(estimated, original, EPS=1e-8):
    target = pow_norm(estimated, original) * original / (pow_p_norm(original) + EPS)
    noise = estimated - original
    sdr = 10 * torch.log10(pow_p_norm(target) / (pow_p_norm(noise) + EPS) + EPS)
    return torch.mean(sdr)

