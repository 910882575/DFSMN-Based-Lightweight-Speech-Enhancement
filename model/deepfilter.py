#!/usr/bin/env python
# coding=utf-8
''' 
Written by yxhu@NPU-ASLP in Tencent AiLAB on 2020.8
arrowhyx@foxmail.com
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
class DeepFilter(nn.Module):
    def __init__(self, L, I):
        '''
        inputs would be [B,D,T]
        L in paper is for time dimension 
        I in paper is for freq dimension 
        '''
        super(DeepFilter, self).__init__()
        self.L = L
        self.I = I
        t_width = L*2+1
        f_width = I*2+1
        kernel = torch.eye(t_width*f_width)
        self.register_buffer('kernel', torch.reshape(kernel, [t_width*f_width, 1, f_width, t_width]))       
    def forward(self, inputs, filters):
        '''
            inputs is [real, imag]: [ [B,D,T], [B,D,T] ] 
            filters is [real, imag]: [ [B,D,T], [B,D,T] ] 
        '''
        # to [B, (2*L+1)*(2*I+1), D, T]
        chunked_inputs = F.conv2d(
                                    torch.cat(inputs,0)[:,None],
                                    self.kernel, 
                                    padding= [self.I, self.L],
            )   
        inputs_r, inputs_i = torch.chunk(chunked_inputs, 2, 0)
        # to [B, (2*L+1)*(2*I+1), D, T]
        chunked_filters = F.conv2d(
                                    torch.cat(filters,0)[:,None],
                                    self.kernel, 
                                    padding= [self.I, self.L],
            )   
        filters_r, filters_i = torch.chunk(chunked_filters, 2, 0)
        outputs_r = inputs_r*filters_r - inputs_i*filters_i
        outputs_i = inputs_r*filters_i + inputs_r*filters_i
        # to [B, D, T]
        outputs_r = torch.sum(outputs_r, 1)
        outputs_i = torch.sum(outputs_i, 1)
        return torch.cat([outputs_r,outputs_i],1) 

if __name__ == '__main__':
    inputs = [
                torch.randn(10,256,99),
                torch.randn(10,256,99),
    ]
    mask = [
                torch.randn(10,256,99),
                torch.randn(10,256,99),
    ]
    net = DeepFilter(1,5)
    outputs =  net(inputs, mask)
    print(outputs.shape)
