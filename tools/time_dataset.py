#!/usr/bin/env python
# coding=utf-8
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
from misc import read_and_config_file
import multiprocessing as mp
from utils import audioread, audiowrite

class DataReader(object):
    def __init__(self, file_name, sample_rate=16000):
        self.file_list = read_and_config_file(file_name, decode=True)
        self.sample_rate = sample_rate 

    def extract_feature(self, path):
        path = path['inputs']
        utt_id = path.split('/')[-1]
        data, fs= audioread(path) 
        if fs != self.sample_rate:
            raise Warning("file {:s}'s sample rate is not match {:d}!".format(path, self.sample_rate)) 
        inputs = np.reshape(data, [1, data.shape[0]]).astype(np.float32)
        
        return inputs, utt_id, data.shape[0]
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return self.extract_feature(self.file_list[index])


class Processer(object):

    def __init__(self, ):
        pass 

    def process(self, path, start_time, segement_length):
         
        wave_inputs = audioread(path['inputs'])[0]
        wave_s1 = audioread(path['labels'])[0]
        if start_time == -1:
            wave_inputs = np.concatenate([wave_inputs, wave_inputs[:segement_length-wave_inputs.shape[0]]])
            wave_s1 = np.concatenate([wave_s1, wave_s1[:segement_length-wave_s1.shape[0]]])
        else:
            wave_inputs = wave_inputs[start_time:start_time+segement_length]
            wave_s1 = wave_s1[start_time:start_time+segement_length]
        
        # I find some sample are not fixed to segement_length,
        # so i padding zero it into segement_length
        if wave_inputs.shape[0] != segement_length:
            padded_inputs = np.zeros(segement_length, dtype=np.float32)
            padded_s1 = np.zeros(segement_length, dtype=np.float32)
            padded_inputs[:wave_inputs.shape[0]] = wave_inputs
            padded_s1[:wave_s1.shape[0]] = wave_s1
        else:
            padded_inputs = wave_inputs
            padded_s1 = wave_s1

        return padded_inputs, padded_s1

class TFDataset(Dataset):

    def __init__(
            self,
            scp_file_name,
            segement_length=8,
            sample_rate=16000,
            processer=Processer(),
            gender2spk=None
        ):
        '''
            scp_file_name: the list include:[input_wave_path, output_wave_path, duration]
            spk_emb_scp: a speaker embedding ark's scp 
            segement_length: to clip data in a fix length segment, default: 4s
            sample_rate: the sample rate of wav, default: 16000
            processer: a processer class to handle wave data 
            gender2spk: a list include gender2spk, default: None
        '''
        self.wav_list = read_and_config_file(scp_file_name)
        self.processer = processer
        mgr = mp.Manager()
        self.index =mgr.list()#[d for b in buckets for d in b]
        self.segement_length = segement_length * sample_rate
        self._dochunk(SAMPLE_RATE=sample_rate)


    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_info, start_time = self.index[index]
        inputs, s1 = self.processer.process({'inputs':data_info['inputs'],'labels':data_info['labels']}, start_time, self.segement_length)
        return inputs, s1

    def _dochunk(self, SAMPLE_RATE=16000, num_threads=12):
        # mutliproccesing
        def worker(target_list, result_list, start, end, segement_length, SAMPLE_RATE):
            for item in target_list[start:end]:
                duration = item['duration']
                length = duration*SAMPLE_RATE
                if length < segement_length:
                    if length * 2 < segement_length:
                        continue
                    result_list.append([item, -1])
                else:
                    sample_index = 0
                    while sample_index + segement_length < length:
                        result_list.append(
                            [item, sample_index])
                        sample_index += segement_length
                    if sample_index != length - 1:
                        result_list.append([
                            item,
                            int(length - segement_length),
                        ])
        pc_list = []
        stride = len(self.wav_list) // num_threads
        if stride < 100:
            p = mp.Process(
                            target=worker,
                            args=(
                                    self.wav_list,
                                    self.index,
                                    0,
                                    len(self.wav_list),
                                    self.segement_length,
                                    SAMPLE_RATE,
                                )
                        )
            p.start()
            pc_list.append(p)
        else: 
            for idx in range(num_threads):
                if idx == num_threads-1:
                    end = len(self.wav_list)
                else:
                    end = (idx+1)*stride
                p = mp.Process(
                                target=worker,
                                args=(
                                    self.wav_list,
                                    self.index,
                                    idx*stride,
                                    end,
                                    self.segement_length,
                                    SAMPLE_RATE,
                                )
                            )
                p.start()
                pc_list.append(p)
        for p in pc_list:
            p.join()
            p.terminate()

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
    inputs, s1 = zip(*data)
    inputs = np.array(inputs, dtype=np.float32)
    s1 = np.array(s1, dtype=np.float32)
    return torch.from_numpy(inputs), torch.from_numpy(s1)

def make_loader(scp_file_name, batch_size, segement_length=8,num_workers=12, processer=Processer()):
    dataset = TFDataset(scp_file_name, segement_length, processer=processer)
    sampler = Sampler(dataset, batch_size)
    loader = tud.DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            sampler=sampler,
                            drop_last=False
                        )
                            #shuffle=True,
    return loader, None #, Dataset
if __name__ == '__main__':
    laoder,_ = make_loader('../data/dev_aishell1_-5~20.lst', 32, num_workers=16)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import time 
    #import soundfile as sf
    stime = time.time()

    for epoch in range(10):
        for idx, data in enumerate(laoder):
            inputs, labels= data 
            inputs.cuda()
            labels.cuda()
            if idx%100 == 0:
                etime = time.time()
                print(epoch, idx, labels.size(), (etime-stime)/100)
                stime = etime
