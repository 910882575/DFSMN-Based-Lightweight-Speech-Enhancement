#!/usr/bin/env python
# coding=utf-8
'''

yxhu@NPU-ASLP in Sogou inc.

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
sys.path.append(os.path.dirname(sys.path[0])+'/tools/speech_processing_toolbox/')
sys.path.append('/home/work_nfs3/yxhu/workspace/se-cldnn-torch/tools/speech_processing_toolbox/')
sys.path.append(
    os.path.dirname(__file__) + '/tools/speech_processing_toolbox/')
sys.path.append(
    os.path.dirname(__file__))
import soundfile as sf
import voicetool.base as voicebox
from misc import read_and_config_file
import multiprocessing as mp
from Augmentation import  MixSpeaker, AddNoise
from drc import drc

class DataReader(object):
    def __init__(self, file_name,adaptation_wavs_dir, win_len=400, win_inc=100,left_context=0,right_context=0, fft_len=512, window_type='hamming', target_mode='MSA',sample_rate=16000):
        self.left_context = left_context
        self.right_context = right_context
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.window = {
                        'hamming':np.hamming(self.win_len)/1.2607934,
                        'hanning':np.hanning(self.win_len),
                        'none':np.ones(self.win_len)
                      }[window_type]
        self.file_list = read_and_config_file(file_name, decode=True)
        self.label_type = target_mode
        self.adaptation_wavs_dir = adaptation_wavs_dir

    def extract_feature(self, path):

        # process input feature
        wave_path = path['inputs']
        utt_id = wave_path.split('/')[-1]
        data = voicebox.audioread(wave_path)

        # Get embedding 
        emb_wave_path = path['emb'] 
        # just for Aishell-1 
        if not os.path.isfile(emb_wave_path):
            spk_dir = emb_wave_path[6:11]
            emb_wave_path = os.path.join(self.adaptation_wavs_dir, spk_dir, emb_wave_path+'.wav')
     
        inputs = data[None,:].astype(np.float32) 
        emb = voicebox.audioread(emb_wave_path).astype(np.float32)
        #emb = drc(emb, 16000, 15,0.03,0.98)
        emb = np.reshape(emb, [1, -1]) 
        nsamples = data.shape[0]
        return inputs, emb, np.angle(inputs), utt_id, nsamples

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return self.extract_feature(self.file_list[index])

class FixDataset(Dataset):

    def __init__(
            self,
            scp_file_name,
            emb_wav_path,
            segement_length=8,
            sample_rate=16000,
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
        mgr = mp.Manager()
        self.index =mgr.list()#[d for b in buckets for d in b]
        self.segement_length = segement_length * sample_rate 
        self.emb_wave_path = emb_wav_path
        _dochunk(self.wav_list, self.index, self.segement_length) 

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_info, start_time = self.index[index]
        emb_wave_path = find_emb_path(data_info['emb'], self.emb_wave_path)
        inputs = clip_data(data_info['path']['inputs'], start_time, self.segement_length)
        labels = clip_data(data_info['path']['labels'], start_time, self.segement_length)
        emb = clip_data(emb_wave_path, -2, self.segement_length)
        return inputs, labels, emb

def clip_data(path, start, segement_length):
    data, fs = sf.read(path)
    if start == -2:
        if data.shape[0] > segement_length:
            start = 0
        else:
            start = -1
    if start == -1:
        data = np.concatenate([data, data[:segement_length-data.shape[0]]])
    else:
        data = data[start:start+segement_length]
    if data.shape[0] < segement_length:
        data = np.pad(data, [0,segement_length-data.shape[0]])
    return data

def find_emb_path(cond, emb_wave_path):
    idx = 0
    while True:
        # Attention and Warning !!!!!!!!
        # This is just for aishell-1
        # Please make sure the adaptation wav's path !!! 
        # start 
        emb_name = random.sample(cond,1)[0]
        if not os.path.isfile(emb_name):
            spk_dir = emb_name[6:11]
            if spk_dir[0] !='S':
                spk_dir = emb_name.split('_')[0]
            emb_path = emb_wave_path+'/'+spk_dir+'/'+emb_name+".wav"
        else:
            emb_path = emb_name
        idx+=1
        data, fs = sf.read(emb_path)
        #end
        if data.shape[0] > 3*fs:
        # chose not too short
            break
        if idx >=100:
            raise RuntimeError('can not find a useable embedding!!!')
    return emb_path

class AutoDataset(Dataset):

    def __init__(
            self,
            target_scp,
            infer_scp,
            noise_scp,
            segement_length=10,
            repeate=4,
            sample_rate=16000,
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
        self.loadscp(target_scp)
        mgr = mp.Manager()
        self.index =mgr.list()#[d for b in buckets for d in b]
        self.segement_length = segement_length * sample_rate 
        _dochunk(self.wav_list, self.index, self.segement_length)
        self.index *= repeate
        self.augmentation = MixSpeaker(infer_scp, snr_range=[0,30], mode='first')
        self.addnosie = AddNoise(noise_scp, snr_range=[12,30])
        self.randstates = [ np.random.RandomState(idx) for idx in range(3000)]

    def loadscp(self, scp_list):
        self.wav_list = [] 
        with open(scp_list) as fid:
            for line in fid:
                tmp = line.strip().split()
                spkid = tmp[0]
                path = tmp[1]
                data, fs = sf.read(path)
                duration = data.shape[0]
                self.wav_list.append({'spkid':spkid, 'path':tmp[1], 'duration':duration})

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_info, start_time = self.index[index]
        spkid, path = data_info['spkid'], data_info['path']
        Y, X, emb, snrs = self.augmentation(
                                    (spkid, path),
                                    start=start_time,
                                    randstat=self.randstates[index%3000],
                                    segement_length=self.segement_length
                                )
        tmp = self.addnosie(Y, 
                            randstat=self.randstates[(index+11)%3000])
        Y = tmp[0]
        X = tmp[3]*X[0]
        tmp = self.addnosie(emb, randstat=self.randstates[(index+17)%3000])
        emb = tmp[0]
        inputs = Y
        labels = X

        return inputs, labels, emb

def worker(target_list, result_list, start, end, segement_length):
    for item in target_list[start:end]:
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
            result_list.append([item, sample_index])
        else:
            sample_index = 0
            while sample_index + segement_length < length:
                result_list.append(
                        [item, sample_index])
                sample_index += segement_length

            if sample_index <= length:
                    result_list.append([
                        item,
                        int(length - segement_length),
                ])
                


def _dochunk(wav_list, index, segement_length=16000*4, num_threads=12):
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
    #inputs, labels, embeddings, snrs = zip(*data)
    inputs, labels, embeddings = zip(*data)
    inputs = np.array(inputs, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    embeddings = np.array(embeddings, dtype=np.float32)
    return torch.from_numpy(inputs),\
            torch.from_numpy(labels),\
            torch.from_numpy(embeddings)


def make_fix_loader(scp_file_name, spk_emb_scp, batch_size, num_workers=12,):
    dataset = FixDataset(scp_file_name, spk_emb_scp,)
    sampler = Sampler(dataset, batch_size)
    loader = tud.DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            sampler=sampler,
                            drop_last=False
                        )
    return loader, None

def make_auto_loader(target_scp, infer_scp, noise_scp, batch_size, num_workers=12, repeate=4):
    dataset = AutoDataset(target_scp, infer_scp, noise_scp, repeate=repeate)
    sampler = Sampler(dataset, batch_size)
    loader = tud.DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            sampler=sampler,
                            drop_last=False
                        )
                            #shuffle=True,
    return loader, None #dataset


if __name__ == '__main__':
    torch.manual_seed(20)
    test_mode = 1
    if test_mode == 1:
        laoder, dataset = make_auto_loader(
                            '/search/odin/huyanxin/data/sogou_train/t1.scp',
                            '/search/odin/huyanxin/data/sogou_train/t1.scp',
                            #'/search/odin/huyanxin/data/sogou_train/spk_list/train_target_spk2wav.scp', 
                            #'/search/odin/huyanxin/data/sogou_train/spk_list/train_spk2wav.scp',
                            '/search/odin/huyanxin/workspace/voicefilter-yxhu/data/musan_train.scp',
                            32, num_workers=16)
    else:
        laoder,_ = make_fix_loader(
                        '../data/dev.lst', '/search/odin/huyanxin/data/data_aishell/all_wavs/',
                            800, num_workers=16)
    import time
    stime = time.time()
    fid = open('log.csv', 'w') 
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
            inputs = data 
            if (idx+1)%5 == 0:
                etime = time.time()
                print(epoch, idx, inputs[0].size(), (etime-stime)/100-1.)
                idx=str(idx)
                epoch=str(epoch)
                sf.write(epoch+'_'+idx+'_noisy.wav', inputs[0][0].numpy(), 16000) 
                sf.write(epoch+'_'+idx+'_clean.wav', inputs[1][0].numpy(), 16000) 
                sf.write(epoch+'_'+idx+'_emb.wav',  inputs[2][0].data.numpy(), 16000) 
                break
                stime = etime

