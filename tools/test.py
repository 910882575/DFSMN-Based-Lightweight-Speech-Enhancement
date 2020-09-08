#!/usr/bin/env python
# coding=utf-8

import scipy.signal as ss
import soundfile as sf
import numpy as np

def addRir(data, rirs, fs, predelay=200, mono=True):
    '''
        data, rirs: [length,nmic]
        predelay: ms
    '''
    length = data.shape[0]
    
    if mono:
        if len(rirs.shape) > 1:
            rirs = rirs[:,0]
        if len(data.shape) > 1:
            data = data[:,0]
    else:
        rirs = rirs.T
        if len(data.shape) > 1:
            data = data[:,0]
        data = data[None,:] 

    reverb = ss.fftconvolve(
            data, rirs,
            mode='full',
        )
    # https://github.com/nttcslab-sp/dnn_wpe/blob/master/example/dataset.py#L393
    dt = np.argmax(rirs,axis=-1).min()
    # early rev + direct 
    et = dt + int(predelay*fs/1000 )
    if mono:
        et_rirs = rirs[:et]
    else:
        et_rirs = rirs[:,:et]
    direct = ss.fftconvolve(
                data,et_rirs,
                mode='full'
            )
    if mono:
        reverb = reverb[:length]
        direct = direct[:length]
    else:
        reverb = reverb[:,:length]
        direct = direct[:,:length]
    residual = reverb - direct 
    return reverb, direct, residual

def strip(data, fs, threshold=6e-4):
    window = 5*fs//1000
    stride = window//2
    t = np.abs(data)
    idx = int(0.2*fs)//stride
        
    while idx*stride + window <= data.shape[0]:
        energy = np.mean(t[idx*stride:idx*stride+window])        
        if energy < threshold:
            return data[:idx*stride]     
        idx+=1
    return data 

def test():
    data = np.random.randn(16000*4,3)
    rirs = np.random.randn(16000//2,4)
    result = addRir(data, rirs, 16000, mono=True)[0]
    print(result.shape)
    result = addRir(data[:,0], rirs, 16000, mono=True)[0]
    print(result.shape)
    result = addRir(data, rirs, 16000, mono=False)[0]
    print(result.shape)
    
    result = addRir(data[:,0], rirs, 16000, mono=False)[0]
    print(result.shape)


def test2():
    
    data,fs = sf.read('./test.wav')
    rir, fs = sf.read('./air_type1_air_binaural_aula_carolina_1_3_135_3.wav')
    rir = strip(rir,fs)
        
    reverb, direct, residual = addRir(data,rir,fs)
    
    print(np.mean((reverb-residual)**2))
    sf.write('./reverb.wav', reverb, fs)
    sf.write('./direct.wav', direct, fs)
    sf.write('./residual.wav', residual, fs)

test2()
