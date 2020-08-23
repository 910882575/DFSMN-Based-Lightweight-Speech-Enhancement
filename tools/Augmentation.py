'''

yxhu@NPU-ASLP in Sogou inc.

'''




import torch
import numpy as np
import soundfile as sf
import os
import sys
import scipy
import multiprocessing as mp
import time
import warnings

eps = np.finfo(np.float32).eps

def audioread(inputs):
    data, fs = sf.read(inputs)
    if len(data.shape) > 1:
        data = data[:,0]
    return data, fs

def activelev(data):
    # normalized to 0db
    max_val = (1. + eps) /( np.std(data) + eps)
    data = data * max_val
    return data

def loadSCP(file_path):
    with open(file_path) as fid:
        result = []
        for line in fid:
            tmp = line.strip().split()
            item = {}
            if len(tmp) == 1:
                item['path'] = tmp[0]
            elif len(tmp) == 2:
                # spkid wave_path
                item['spkid'] =  tmp[0]
                if os.path.isfile(tmp[1]):
                    item['path'] = tmp[1] 
                else:
                    item['gender'] = tmp[1]
            result.append(item)             
    #        elif len(tmp) == 3:
    #            # spkid gender wave_path
    #            item['spkid'] =  tmp[0]
    #            item['path'] = tmp[2]
    #            item['gender'] = tmp[3]
    #        result.append(item)
    #if len(result[0].keys()) == 3:
    #    spk2wav = {}
    #    spk2gender = {}
    #    for item in result:
    #        spk2wav[item['spkid']] = item['path']
    #        spk2gender[item['spkid']] = item['gender']
    #    return spk2wav, spk2gender
    return result


class Augmentation(object):

    def __init__(self, wav_list):
        pass             
    
    def __call__(self, inputs, fs=None):
        
        if isinstance(inputs, str):
            data, fs = audioread(inputs)
            return self.fit(data, fs)
        else:
            return self.fit(inputs, fs)

    def fit(self, inputs, fs):
        pass 

def mixnoise(inputs, inferences, start, segement_length, snrs, scale, randstat):
    if isinstance(inputs, np.ndarray):
        signal = inputs
    elif isinstance(inputs, str):
        signal = sf.read(inputs)[0]
    else:
        raise RuntimeError("In mixup: Please send a valid inputs, path (str) or data (np.array)")
    
    #if start is not None and segement_length is not None:
   #     if start == -1:
   #         signal = np.concatenate([signal, signal[:segement_length-signal.shape[0]]])
   #     else:
   #         signal=signal[start:start+segement_length]
    # s1/s2 s1/s2 s1/s3
    try:
        signal_len = signal.shape[0]
        infer = np.zeros(signal_len)
        if isinstance(inferences, list):
            for path, snr in zip(inferences, snrs):
                noise_data, fs = sf.read(path)
                if noise_data.shape[0] != signal_len:
                    st = randstat.randint(np.abs(noise_data.shape[0]-signal_len))
                else:
                    st = 0
                if noise_data.shape[0] > signal_len:
                    noise = noise_data[st:st+signal_len]
                    weight = np.sqrt(np.var(noise)/10**(snr/10))
                    infer += noise*weight
                else:
                    weight = np.sqrt(np.var(noise_data)/10**(snr/10))
                    infer[st:st+noise_data.shape[0]] += noise_data*weight
        else:
            noise_data, fs = audioread(inferences)
            st = randstat.randint(np.abs(noise_data.shape[0]-signal_len))
            if noise_data.shape[0] > signal_len:
                noise = noise_data[st:st+signal_len]
                infer += noise
            else:
                infer[st:st+noise_data.shape[0]] += noise_data*weight
        if isinstance(snrs, list):
            infer_weight = np.sqrt(np.var(signal)/np.var(infer)/10**(snrs[0]/10))
        else:
            infer_weight = np.sqrt(np.var(signal)/np.var(infer)/10**(snrs/10))
        infer = infer_weight * infer
        mix = signal + infer
        mix_fac = 1./np.max([np.abs(mix), np.abs(signal), np.abs(infer)])*scale
        mix = mix*mix_fac
        signal = signal*mix_fac
        infer = infer*mix_fac
    except Exception as e:
        print(signal.shape, noise_data.shape, '!!!!!!!!!!!!!', randstat)
    return mix, signal, infer, mix_fac

def clip_data(data, start, segement_length):
    tgt = np.zeros(segement_length)*1e-4
    data_len = data.shape[0]
    if start == -2:
        # this means  segement_length//4 < data_len < segement_length//2
        # padding to A_A_A
        if data_len < segement_length//3:
            data = np.pad(data, [0,segement_length//3-data_len])
            tgt[:segement_length//3] += data 
            st = segement_length//3
            tgt[st:st+data.shape[0]] += data
            st = segement_length//3*2
            tgt[st:st+data.shape[0]] = data

        else:
            st = (segement_length//2-data_len)%101
            tgt[st:st+data_len] += data
            st = segement_length//2+(segement_length//2-data_len)%173
            tgt[st:st+data_len] += data

    elif start == -1:
        # this means  segement_length < data_len*2
        # padding to A_A 
        if data_len %4 == 0:
            tgt[:data_len] += data
            
            tgt[data_len:] += data[:segement_length-data_len]
        elif data_len %4 == 1:
            tgt[:data_len] += data
        elif data_len %4 == 2:
            tgt[-data_len:] += data
        elif data_len %4 == 3:
            tgt[(segement_length-data_len)//2:(segement_length-data_len)//2+data_len] += data

    else:
        # this means  segement_length < data_len
        tgt += data[start:start+segement_length]
    return tgt


def mixspeech(speeches, snrs, scale, mode, start, segement_length):
    wavs = []
    max_len = 0
    min_len = 1e12
    for idx, spk in enumerate(speeches):

        if isinstance(spk, str):
            data, fs = audioread(spk)
        else:
            data = spk 
        if mode == 'first' and  idx == 0: 
            data = clip_data(data, start, segement_length) 
        wavs.append(data)
        if max_len < data.shape[0]:
            max_len = data.shape[0] 
        if min_len > data.shape[0]:
            min_len = data.shape[0]

    if mode == 'min':
        mix = np.zeros(min_len)
    elif mode=='max': # max 
        mix = np.zeros(max_len)
    elif mode == 'first':
        max_len = wavs[0].shape[0]
        mix = np.zeros(wavs[0].shape[0])

    processed_wavs = []
    idx = 0
    for data, snr in zip(wavs,snrs):
        data_len = data.shape[0]
        if mode == 'min':
            st = np.random.randint(np.abs(data_len - min_len)+1)
            data = activelev(data[st:st+min_len])

        elif mode == 'max': # max 
            if max_len == data_len:
                st = 0
            else:
                st = np.random.randint(max_len - data_len)
            data = activelev(data)
            
            weight = 10**(snr/40)
            data = data * weight
            if st > 0:
                data_t = np.zeros(max_len)
                data_t[-st:-st+data_len] = data 

        elif mode == 'first':
            if max_len > data_len:
                st = np.random.randint(max_len - data_len)
                data_t = np.zeros(max_len)
                data_t[st:st+data_len] = data
                data = data_t
            elif max_len < data_len:
                st = np.random.randint(data_len - max_len)
                data = data[st:st+max_len]
            data = activelev(data)
        
        weight = 10**(snr/40)
        data = data * weight
        np.random.seed(np.abs(int(snr))+1)
        if idx == -1:
            white = np.random.randn(data.shape[0])
            white = np.clip(white, -1, 1)
            max_t = np.max(np.abs(data)) 
            white = max_t*0.003*white
            data += white
            idx +=1
        mix += data
        processed_wavs.append(data)
    mix_fac = 1./np.max(np.abs([mix]+processed_wavs))*scale
    mix *= mix_fac
    processed_wavs = [ x*mix_fac for x in processed_wavs]
    return mix, processed_wavs, mix_fac

class MixSpeaker(object):
    def __init__(self, input_list, mix_num=2, snr_range=[-5,5], mode='min'):
        
        self.snr_range = snr_range
        if isinstance(input_list, str):
            if input_list.endswith('list') or input_list.endswith('lst'):
                # mix list should include:
                #               speech1 snr1 speech2 snr2 speech3 snr3... scale
                self.mix_list = []
                with open(input_list) as fid:
                    for line in fid:
                        tmp = line.strip().split()
                        tlen = len(tmp)
                        speechs, snrs = [], []
                        scale = float(tmp[-1])
                        for idx in range(0, tlen-1, 2):
                            speechs.append(tmp[idx])
                            snrs.append(float(tmp[idx+1]))
                        self.mix_list.append([ speechs, snrs, scale])
                self.spk2wav = None
                self.len = len(self.mix_list)
                self.num_spk = mix_num + 2
            elif input_list.endswith('scp'):
                # spk2wav scp should include:
                #               spkid wave_path
                # spk2gender scp should include:
                #               spkid gender
                self.mix_list = None
#                self.spk2wav, self.spk2gender = loadSCP(input_list)
                tmp = loadSCP(input_list)
                self.spk2wav = {}
                self.spk2gender = {}
                for item in tmp:
                    spkid = item['spkid']
                    path = item['path']
                    if spkid in self.spk2wav:
                        self.spk2wav[spkid].append(path)
                    else:
                        self.spk2wav[spkid] = [path]
                        self.spk2gender[spkid] = 'M'
                self.spks = list(self.spk2wav.keys())
                self.num_spk = len(self.spks)
                self.len = self.num_spk

        elif isinstance(input_list, list):
            self.mix_list = None
            # this means, input_list is [spk2wav, spk2gender]
            self.spk2wav = {}
            self.spk2gender = {}
            tmp = loadSCP(input_list[0])
            for item in tmp:
                spkid = item['spkid']
                path = item['path']
                if spkid in self.spk2wav:
                    self.spk2wav[spkid].append(path)
                else:
                    self.spk2wav[spkid] = [path]
            tmp = loadSCP(input_list[1])
            for item in tmp:
                self.spk2gender[item['spkid']] = item['gender']
            self.spks = list(self.spk2wav.keys())
            self.num_spk = len(self.spks)
            self.len = self.num_spk

        self.mix_num = mix_num
        self.mode = mode

        if self.mix_num > self.num_spk:
            warnings.warn(
                'the num of target mix speakers > overall speakers! '
                'Are you sure?!'
                )
   
    def __len__(self):
        return self.len

    def __call__(self, inputs, randstat=None, start=None, segement_length=None, rescale=True):
        '''
            inputs is int, mix speech with fixed configeration eg: mix_list
            inputs is str, online mix speech
                    if inputs == 'random': random select mix_num's and mix
                    else inputs should be [spkid, wave_path] : mix this wave with random select (mix_num - 1) speech
        '''
    
        if isinstance(inputs, str) and self.spk2wav is None:
            raise RuntimeError("Please set inputs a value"
                               "and MixSpeaker.input_list to mix_scp!!!!!!!")

        if not isinstance(inputs, int):
            if randstat is None:
                randstat = np.random.RandomState(int(time.time())%1579)
            spks = []
            wavs = []
            snrs = []
            genders = []
            target_spk = None
            target_wave_path = None
            if inputs != 'random':
                # mix target wav with (mix_num -1)'s wavs
                target_spk, target_wave_path = inputs
            elif inputs == 'random':
                idx = randstat.randint(self.num_spk)
                target_spk = self.spks[idx]
                idx = randstat.randint(len(self.spk2wav[target_spk]))
                target_wave_path = self.spk2wav[target_spk][idx]

            # get embedding
            emb_path = '' 
            tick = 0
            emb_seg_length=segement_length*8//12
            while True:
                idx = randstat.randint(len(self.spk2wav[target_spk]))
                emb_path = self.spk2wav[target_spk][idx]
                emb = sf.read(emb_path)[0]
                if emb_path != target_wave_path and emb.shape[0] > emb_seg_length//3:
                    break
                if tick >= 100:
                    raise RuntimeError('Can not find embedding wav path') 

            if emb.shape[0] > emb_seg_length:
                st = randstat.randint(emb.shape[0] - emb_seg_length)
                emb = emb[st:st+emb_seg_length]
            else:
                st = emb_seg_length - emb.shape[0]
                emb = np.concatenate([emb, emb[:st]])
                if emb_seg_length > emb.shape[0]:
                    emb = np.pad(emb, [0, emb_seg_length-emb.shape[0]])
            
            wavs.append(target_wave_path)
            spks.append(target_spk)
            genders.append(self.spk2gender[target_spk])

            # 1. select other spker's wav 
            for counter in range(self.mix_num-1):
                tick = 0
                while True:
                    # get one spker
                    idx = randstat.randint(self.num_spk)
                    target_spk = self.spks[idx]
                    
                    # filter the repeat speaker
                    flag = True
                    for item in spks:
                        if target_spk == item:
                            flag = False
                    if flag:
                        # select one wav of this speaker
                        idx = randstat.randint(len(self.spk2wav[target_spk]))
                        wave_path = self.spk2wav[target_spk][idx]
                        # append to list
                        wavs.append(wave_path)
                        spks.append(target_spk)
                        genders.append(self.spk2gender[target_spk])
                        break
                    tick+=1
                    if tick >= 100:
                        raise RuntimeError('Can not find other spker to be inference') 

            for counter, spkid in enumerate(spks):
                
                # random generator snr in self.snr_range
                if counter == 0:
                    snr = randstat.uniform(self.snr_range[0], self.snr_range[1])
                elif counter ==1:
                    snr = -snrs[0]
                else:
                    snr = 0.
                snrs.append(snr)
            
            if rescale == True:
                scale = randstat.normal()*0.5 + 0.9
                lower = 0.35
                upper = 0.95
                if scale< lower or scale > upper:
                    scale = randstat.uniform(lower, upper) 
            else:
                scale = 1.
    
            mix, spks, fac = mixspeech(wavs, snrs, scale, self.mode, start, segement_length)
            return mix, spks, emb, snrs

        else:
            if self.mix_list is None:
                raise RuntimeError("Please set inputs a value"
                               "and AddNoise.input_list to mix_speaker.list !!!!!!!")
            spks, snrs, scale = self.mix_list[inputs]
            assert(len(spks) == len(snrs))
            mix, spks, fac = mixspeech(spks, snrs, scale, self.mode, start, segement_length)
            return mix, spks, fac 


class AddNoise(object):

    def __init__(self, input_list, noise_type=1, snr_range=[-5, 20]):
        '''
        input_list:
            if input_list's suffix is 'lst' or 'list':
                it means a mix list, will run in fixed addnosie mode
            elif it's suffix is 'scp':
                it means a wav.scp, and will run in online addnosie mode
        noise_type:
            how many noise will be added to speech

        snr_range:
            as it's name
        '''

        if input_list.endswith('list') or input_list.endswith('lst'):

            self.mix_list = [] 
            with open(input_list) as fid:
                for line in fid:
                    tmp = line.strip().split()
                    self.mix_list.append(tmp)
            self.noise_scp = None
            self.len = len(self.mix_list) 
            self.index = mp.Value('i',0)

        elif input_list.endswith('scp'):
            self.noise_scp = loadSCP(input_list)
            self.mix_list = None 
            self.noise_num = len(self.noise_scp)
            self.len = self.noise_num 

        self.noise_type = noise_type
        self.snr_range = snr_range
    
    def __len__(self):
        return self.len
    
    def __call__(self, inputs, randstat=None, start=None, segement_length=None, rescale=True):
        '''
            inputs: int or str
            randstat:
            start:
            segement_length:
            rescale:

        '''
        if isinstance(inputs, str) and self.noise_scp is None:
            raise RuntimeError("Please set inputs a value"
                               "and AddNoise.input_list to noise_scp!!!!!!!")
        if isinstance(inputs, str) or isinstance(inputs, np.ndarray):
            # will add noise online
            if randstat is None:
                randstat = np.random.RandomState(int(inputs[-5])%100)
            if rescale == True:
                scale = randstat.normal()*0.5 + 0.9
                lower = 0.15
                upper = 0.95
                if scale< lower or scale > upper:
                    scale = randstat.uniform(lower, upper)
            else:
                scale = 1.
            noise_names = []
            snrs = []
            for idx in range(self.noise_type):
                rand_id =  randstat.randint(self.noise_num)
                noise_names.append(self.noise_scp[rand_id]['path'])
                snr = randstat.uniform(self.snr_range[0], self.snr_range[1])
                snrs.append(snr)
            Y, X, N, fac = mixnoise(inputs, noise_names, start, segement_length, snrs, scale,randstat)
            return Y, X, N, fac, snrs, scale

        else:
            speech, noise, nosie_start, snr, scale = self.mix_list[inputs]
            snr = float(snr)
            scale= float(scale)
            Y,X,N,fac= mixnoise(speech, noise, start, segement_length, snr, scale, randstat)
            return Y,X,N, fac, snr, scale
            

class Reverberation(object):
    def __init__(self, rir_list):
        pass
def test_addnoise():
    mix_list='./addnoise.lst'
    speech_list='./speech.scp'
    noise_list = './noise.scp'
    print('test online add noise')
    if not os.path.isdir('online_addnoise'):
        os.mkdir('online_addnoise')
    aug = AddNoise(noise_list)
    with open(speech_list) as fid:
        for (idx, line) in enumerate(fid):
            name = line.strip()
            #y,x,n,fac = aug(name)
            y,x,n,f, snr, scale = aug(name)
            print(snr, scale)
            sf.write('online_addnoise/'+str(idx)+'.wav', y, 16000)
    print('test online add noise with mixlist')
    if not os.path.isdir('mix_addnoise'):
        os.mkdir('mix_addnoise')
    aug = AddNoise(mix_list)
    for idx in range(len(aug)):
        y,x,n, fac, snr, scale = aug(idx)
        print(snr,scale)
        sf.write('mix_addnoise/'+str(idx)+'.wav', y, 16000)


def test_mixspeaker():
    mix_list = './mixspeaker.list'
    spk2wav = './speech.scp'
    spk2gender = './spk.scp'
    print('test mix speaker with fixed list')
    if not os.path.isdir('mix_speaker_fixed'):
        os.mkdir('mix_speaker_fixed')
    aug = MixSpeaker(mix_list)
    for idx in range(len(aug)):
        mix, signal, fac = aug(idx)
        sf.write('mix_speaker_fixed/mix_'+str(idx)+'.wav', mix, 16000)
        sf.write('mix_speaker_fixed/s1_'+str(idx)+'.wav', signal[0], 16000)
        sf.write('mix_speaker_fixed/s2_'+str(idx)+'.wav', signal[1], 16000)

    print('test mix speaker for target spker' )
    if not os.path.isdir('mix_target_speaker'):
        os.mkdir('mix_target_speaker')
    
    aug = MixSpeaker([spk2wav, spk2gender])
    with open(spk2wav) as fid:
        idx = 0
        for line in fid:
            tmp = line.strip().split()
            utt_ud = tmp[0]
            name = tmp[1]
            randstat = np.random.RandomState(idx)
            mix, signal, fac = aug([utt_ud, name], randstat)
            sf.write('mix_target_speaker/mix_'+str(idx)+'.wav', mix, 16000)
            sf.write('mix_target_speaker/s1_'+str(idx)+'.wav', signal[0], 16000)
            sf.write('mix_target_speaker/s2_'+str(idx)+'.wav', signal[1], 16000)
            idx += 1
    print('test mix speaker random')
    aug = MixSpeaker(spk2wav)
    if not os.path.isdir('mix_random_speaker'):
        os.mkdir('mix_random_speaker')
    for idx in range(len(aug)):
        randstat = np.random.RandomState(idx)
        mix, signal, fac = aug('random', randstat)
        sf.write('mix_random_speaker/mix_'+str(idx)+'.wav', mix, 16000)
        sf.write('mix_random_speaker/s1_'+str(idx)+'.wav', signal[0], 16000)
        sf.write('mix_random_speaker/s2_'+str(idx)+'.wav', signal[1], 16000)
                  

def test_addrir():
    pass

if __name__ == '__main__':

    test_addnoise()
    test_mixspeaker()
