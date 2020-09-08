#!/usr/bin/env python
# coding=utf-8
'''

yxhu@NPU-ASLP in Sogou inc.
modified by yxhu in Tencent AiLab 2020


'''


import numpy as np
import scipy
import torch 
import random
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data as tud
import os 
import sys
sys.path.append(
    os.path.dirname(__file__))
import soundfile as sf
import multiprocessing as mp
from Augmentation import  Mixer, load_scp, clip_data, load_wav

class AutoDataset(Dataset):

    def __init__(
            self,
            scps,
            rir_scps,
            segement_length=8,
            sample_rate=16000,
            load_memory=False,
            repeate=50,
            mono=False,
        ):
        '''
            scp_file_name: the list include:[input_wave_path, output_wave_path, duration]
            spk_emb_scp: a speaker embedding ark's scp 
            segement_length: to clip data in a fix length segment, default: 4s
            sample_rate: the sample rate of wav, default: 16000
            processer: a processer class to handle wave data 
            gender2spk: a list include gender2spk, default: None
        '''
        mgr = mp.Manager()
        self.mono = mono
        self.index = mgr.list()#[d for b in buckets for d in b]
        self.load_memory = load_memory
        self.sample_rate = sample_rate
        self.wav_list = load_scp(scps[0], load_memory, mono=mono, sample_rate=self.sample_rate)
        print('load target scp success')
        self.segement_length = segement_length * sample_rate 
        _dochunk(self.wav_list, self.index, self.segement_length,sample_rate=self.sample_rate)
        print('chunk data success', len(self.index), len(self.wav_list))
        print(scps)
        np.random.shuffle(self.index)
        self.index *= repeate
        self.mixer = Mixer(mix_scps=[scps[1]], rir_scps=rir_scps, snr_range=[-5,20], mix_mode='first', load_memory=load_memory, mono=mono, sample_rate=sample_rate)
        self.num_states = len(self.index) % 3000
        self.randstates = [ np.random.RandomState(idx*10) for idx in range(self.num_states)]


    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        wav_idx,start_time = self.index[index]
        data_info = self.wav_list[wav_idx]
        spkid, path = data_info['spkid'], data_info['path']
        if self.load_memory:
            path = clip_data(path, start_time, self.segement_length)
        Y, X, D, R, fac,snr,scale = self.mixer(
                                    (spkid, path),
                                    start=start_time,
                                    randstat=self.randstates[index%self.num_states],
                                    segement_length=self.segement_length,
                                    emb_length=18,
                                )
        '''
        X = load_wav(path,self.mono)[0][start_time:start_time+self.segement_length]
        Y = X
        emb =X
        if not self.mono :
            X = np.concatenate(X,-1).T
            Y = Y.T
        else:
            X = np.stack(X,0)
        '''
        return Y, X[0]

def worker(target_list, result_list, start, end, segement_length, sample_rate):
    for idx in range(len(target_list[start:end])):
        item = target_list[idx]
        duration = item['duration']
        length = duration
        if length < segement_length:
            sample_index = -1
            if length * 2 < segement_length and length*4 > segement_length:
                sample_index = -2
            elif length * 2 > segement_length:
                sample_index = -1
            else:
                continue
            result_list.append([idx, sample_index])
        else:
            sample_index = 0 
            if isinstance(item['path'], str):
                data = load_wav(item['path'], sample_rate=sample_rate,mono=True)[0]
            else:
                data = item['path']
            data = np.abs(data)
            threshold=1e-3 # this is just for vocal
            
            while sample_index + segement_length < length:
                # Attention 
                # there is an easy vad to filter 
                # blank
                non_zero = np.sum(data[sample_index:sample_index+segement_length]>threshold)
                
                if non_zero*10 > segement_length:
                    result_list.append(
                        [idx, sample_index])
                    sample_index += (segement_length//3)
                else:
                    #print(segement_length, duration, item['spkid'], non_zero, sample_index)
                    sample_index += segement_length//2

            
            if sample_index < length:
                non_zero = np.sum(data[-length+segement_length:]>threshold)
                if non_zero*4 > segement_length:
                    result_list.append([
                        idx,
                        int(length - segement_length),
                    ])


def _dochunk(wav_list, index, segement_length=16000*4, sample_rate=44100,num_threads=12):
        # mutliproccesing
        pc_list = []
        stride = len(wav_list) // num_threads
        if stride < 100:
            p = mp.Process(
                            target=worker,
                            args=(
                                    wav_list,
                                    index,
                                    0,
                                    len(wav_list),
                                    segement_length,
                                    sample_rate
                                )
                        )
            p.start()
            pc_list.append(p)
        else: 
            for idx in range(num_threads):
                if idx == num_threads-1:
                    end = len(wav_list)
                else:
                    end = (idx+1)*stride
                p = mp.Process(
                                target=worker,
                                args=(
                                    wav_list,
                                    index,
                                    idx*stride,
                                    end,
                                    segement_length,
                                    sample_rate
                                )
                            )
                p.start()
                pc_list.append(p)
        for p in pc_list:
            p.join()

class Sampler(tud.sampler.Sampler):
    '''
     
    '''
    def __init__(self, data_source, batch_size):
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i+batch_size)
                        for i in range(0, it_end, batch_size)]
        self.data_source = data_source
        
    def __iter__(self):
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)


def zero_pad_concat(inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = np.array([len(inputs), max_t, inputs[0].shape[1]])
    inputs_mat = np.zeros(shape, np.float32)
    for idx, inp in enumerate(inputs):
        inputs_mat[idx, :inp.shape[0],:] = inp
    return inputs_mat

def collate_fn(data):
    #inputs, labels, embeddings, spkid= zip(*data)
    inputs, labels = zip(*data)
    inputs = np.array(inputs, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    return torch.from_numpy(inputs),\
           torch.from_numpy(labels)



def make_dataloader(vocals_scp, background_scp, rir_scps, batch_size, sample_rate=16000,num_workers=12, training=True, repeate=10, load_memory=False, segement_length=6, mono=True, sampler=None):
    dataset = AutoDataset(
                            [vocals_scp, background_scp],
                            rir_scps=rir_scps,
                            load_memory=load_memory,
                            repeate=repeate,
                            sample_rate=sample_rate,
                            segement_length=segement_length,
                            mono=mono
                         )
    loader = tud.DataLoader(
                            dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            shuffle=(training and sampler is None),
                            drop_last=False,
                            sampler=sampler
                        )
    return loader


if __name__ == '__main__':
    torch.manual_seed(20)
    test_mode = 0
    if test_mode == 0:
        laoder = make_dataloader(
                            '../data/tr_clean.lst',
                            '../data/tr_noise.lst',
                            '../data/tr_rir.lst',
                            sample_rate=16000,
                            batch_size=32,
                            num_workers=8,
                            training=True,
                            load_memory=False,
                            repeate=30,
                            mono=True
            )
    else:
        laoder= make_dataloader(
                            './vocals',
                            './background',
            32, num_workers=8, training=False)
    import time
    stime = time.time()
    print_freq=10
    for epoch in range(3):
        if epoch == 0 or epoch == 4:
            torch.manual_seed(20)
        else:
            torch.manual_seed(20+epoch)
        '''
        idx=str(99)
        print(dataset)
        epoch=str(epoch)
        inputs = dataset[0][0]
        sf.write(epoch+'_'+idx+'_noisy.wav', inputs, 16000) 
        '''
        print('num_batch', len(laoder))
        for idx, data in enumerate(laoder):
            if (idx+1)%print_freq == 0:
                inputs, labels = data
                etime = time.time()
                print(epoch, idx, inputs.size(), (etime-stime)/print_freq)
                idx=str(idx)
                epoch=str(epoch)
        #        sf.write(epoch+'_'+idx+'_noisy.wav', inputs.numpy().T, 44100) 
        #        sf.write(epoch+'_'+idx+'_clean.wav', labels.numpy().T, 44100) 
        #        sf.write(epoch+'_'+idx+'_emb.wav',  inputs[2].numpy().T, 44100) 
                stime = etime 
        #        break
        break
