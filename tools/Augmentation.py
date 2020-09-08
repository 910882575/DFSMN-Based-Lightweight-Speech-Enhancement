import numpy as np
import soundfile as sf  
import scipy.signal as ss
import librosa
import wave

'''
yxhu@NPU-ASLP in Sogou inc.
modified by yxhu in Tencent AiLab 2020

for music mixer 

'''
eps=1e-8
def activelev(data):
    # normalized to 0db
    nonzero_data =data[np.abs(data)>1e-4]
    if nonzero_data.shape[0] < 100:
        power = 1.
    else:
        power = np.std(nonzero_data)
    
    max_val = 1./(power+1e-4)
    data = data * max_val
    return data

def load_wav(path, sample_rate, mono=False):
    data, fs = sf.read(path)
    if mono and len(data.shape)> 1:
        data = data[:,0]
    if fs != sample_rate :
        #raise RuntimeError("the {:}'s fs is {:d}, which is not match the target fs {:d}".format(path, fs, sample_rate))
        print("the {:}'s fs is {:d}, which is not match the target fs {:d}".format(path, fs, sample_rate))
        data = librosa.resample(data,fs, sample_rate)
        fs = sample_rate
    return data, fs

def get_wave_header(path, sample_rate):

    with wave.open(path, 'rb') as fid:
        nframes = fid.getnframes()
        fs = fid.getframerate()
        duration = nframes/fs*sample_rate
    #print(nframes/fs)
    return int(duration)


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



def load_scp(mix_scp, load_memory, mono, sample_rate):

    result = []
    print(mix_scp)
    with open(mix_scp) as fid:
        for line in fid:
            items = line.strip().split('#')
            if load_memory or True:
                data, fs = load_wav(items[0], sample_rate, mono)
                duration = data.shape[0]
            else:
                duration = get_wave_header(items[0], sample_rate)
                data = None
            item = {
                        'spkid':items[0],
                        'path': data if load_memory else items[0],
                        'spk': items[0],
                        'duration': duration,
                    }
            if len(items)> 2:
                item = dict( item,
                    {
                        str(idx): items[idx] for idx in range(2,len(items))
                    }
                )
            result.append(item
            ) 
    
    return result

def clip_data(data, start, segement_length):
    data_len = data.shape[0]
    shape = list(data.shape)
    shape[0] = segement_length
    tgt = np.zeros(shape)
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
        if tgt.shape[0] != data[start:start+segement_length].shape[0]:
            start = data.shape[0] - segement_length
        tgt += data[start:start+segement_length]
    return tgt


class Mixer(object):

    def __init__(
        self,
        mix_scps=[],
        rir_scps=None,
        mix_nums=2,
        snr_range=(-5,30),
        scale_range=(0.4,0.95),
        load_memory=False,
        mix_mode = 'first',
        sample_rate = 16000,
        mono=False
    ):
        '''
        mix_scps: list, include scps, [scp1, scp2, scp3],
        mix_num: int,  means how many source should be mixed
        snr_range: tuple, include mix snr range
        scale_range: tuple, a scale range, (low, high)
        mix_mode: "min", mixed speech's length equal to min length
              "max", mixed speech's length equal to max length
               "first", mixed speech's length equal to first length
        load_memory: load all data in the memory
        '''

        assert isinstance(mix_scps, list), "mix_scp should be a list include scps: [scp1, scp2, scp3...]"
        self.mix_nums = mix_nums
        self.snr_range=snr_range
        self.scale_range = scale_range
        self.mix_mode = mix_mode
        self.sample_rate = sample_rate
        self.mono = mono
        
        self.mix_scps = [load_scp(scp, load_memory, mono, self.sample_rate) for scp in mix_scps]
        if rir_scps is not None:
            self.rir_scps = load_scp(rir_scps, load_memory, mono, self.sample_rate)
        else:
            self.rir_scps = None #load_scp(rir_scps, load_memory, mono, self.sample_rate)
            

    def _get_scp_idx(self, idx):
        return idx % len(self.mix_scps)

    def _select(self, idx, randstat, filter_spk=None):
        rand_id = -1
        try_times = 100
        while try_times != 0:
            rand_id = randstat.randint(len(self.mix_scps[idx]))
            spk_id = self.mix_scps[idx][rand_id]['spkid']
            if filter_spk is None or spk_id not in filter_spk:
                break
            try_times -= 0
        return rand_id


    def __call__(self, inputs, randstat=None, start=None, segement_length=None, emb_length=15, rescale=True):
        '''
            inputs: str
            randstat:
            start:
            segement_length:
            rescale:
        '''
        if isinstance(inputs, str) and self.mix_scps is None:
            raise RuntimeError("Please set inputs a value"
                               "and AddNoise.input_list to noise_scp!!!!!!!")
        speech = []
        if isinstance(inputs, tuple) or isinstance(inputs, str) or isinstance(inputs, np.ndarray) or inputs is None:
            # will add noise online
            if randstat is None:
                randstat = np.random.RandomState()
            if rescale == True:
                scale = randstat.normal()*0.4 + 0.9
                if scale< self.scale_range[0] or scale > self.scale_range[1]:
                    scale = randstat.uniform(self.scale_range[0], self.scale_range[1])
            else:
                scale = 1.
            snrs = []
            if inputs is not None:
                if isinstance(inputs,tuple):
                    inputs = inputs[1]
                
                if isinstance(inputs,str):
                    inputs, fs = load_wav(
                                        inputs,
                                        self.sample_rate, self.mono)
                speech.append(inputs)
            for idx in range(self.mix_nums):
                if idx != 0 or inputs is None:
                    idx = self._get_scp_idx(idx)
                    counter = 100 
                    while counter > 0:
                        rand_id = self._select(idx, randstat) 
                        length = self.mix_scps[idx][rand_id]['duration'] 
                        if length>2*self.sample_rate:
                            break
                    data, fs = load_wav(
                                        self.mix_scps[idx][rand_id]['path'],
                                        self.sample_rate, self.mono)
                    #speech.append(self.mix_scps[idx][rand_id]['path'])
                    speech.append(data)
                    
                if idx == 0:
                    snr = randstat.uniform(self.snr_range[0], self.snr_range[1])
                elif idx ==1:
                    snr = -snrs[0]
                else:
                    snr = 0.
                snrs.append(snr)
            
            ''' 
            # select from background 
            if isinstance(speech[1], str):
                data, fs = load_wav(speech[1], mono=self.mono)
            else:
                data = speech[1]
            length = np.min([25*self.sample_rate, data.shape[0]])
            seg_len = emb_length*self.sample_rate
            st = randstat.randint(length-seg_len)
            emb = data[st:st+seg_len]
            ''' 
            rirs = None
            direct_speech = None
            residual_speech = None
            reverb_speech = None
            if self.rir_scps is not None:
                rate = randstat.uniform()
                if rate < 0.75:
                    rir_idx = randstat.randint(len(self.rir_scps))
                    rirs, fs = load_wav(self.rir_scps[rir_idx]['path'],self.sample_rate, self.mono)
            
                    clip_speech = clip_data(speech[0], start, segement_length)
                    reverb_speech, direct_speech, residual_speech = addRir(clip_speech, rirs, self.sample_rate,mono=self.mono) 

                    speech[0] = reverb_speech
                
                if rate< 0.3:
                    for idx in range(1,self.mix_nums):
                        rir_idx = randstat.randint(len(self.rir_scps))
                        rirs, fs = load_wav(self.rir_scps[rir_idx]['path'],self.sample_rate, self.mono)
                        other = speech[idx]#clip_data(speech[idx], start, segement_length)
                        reverb_other, direct_other, residual_other = addRir(other, rirs, self.sample_rate,mono=self.mono)
                        speech[idx] = reverb_other
            
            #rand_id = randstat.randint(len(self.mix_scps[idx]))
            M, S, fac = mixspeech(speech, snrs, scale, self.mix_mode, start, \
                                  segement_length, randstat, self.mono, self.sample_rate)
            '''
            max_fac = np.max(np.abs(emb))
            if max_fac >1e-1:
                emb = emb/max_fac*scale
            emb=None
            '''
            if reverb_speech is not None:
                #reverb_speech *= fac 
                max_v = np.max(S[0])
                max_r = np.max(reverb_speech) 
                reverb_speech = reverb_speech/max_r*max_v
                residual_speech = residual_speech/max_r*max_v

            return M, S, reverb_speech, residual_speech, fac, snrs, scale



def mixspeech(speeches, snrs, scale, mode, start, segement_length, randstat, mono, sample_rate):
    wavs = []
    max_len = 0
    min_len = 1e12
    max_p = -1
    for idx, spk in enumerate(speeches):

        if isinstance(spk, str):
            data, fs = load_wav(spk, sample_rate, mono)
        else:
            data = spk 
        if mode == 'first' and  idx == 0 and data.shape[0] != segement_length: 
            data = clip_data(data, start, segement_length)  
        '''
        if idx != 0:
            data = data[segement_length*4:]
        '''
        wavs.append(data)
        if max_len < data.shape[0]:
            max_len = data.shape[0] 
        if min_len > data.shape[0]:
            min_len = data.shape[0]
    shape = list(wavs[0].shape)
    if mode == 'min':
        shape[0] = min_len
    elif mode=='max': # max 
        shape[0] = max_len
    elif mode == 'first':
        shape[0] = wavs[0].shape[0]
        max_len = shape[0]
    mix = np.zeros(shape)

    processed_wavs = []
    add_white = False
    idx = 0
    for data, snr in zip(wavs,snrs):
        data_len = data.shape[0]
        if mode == 'min':
            st = randstat.randint(np.abs(data_len - min_len)+1)
            data = activelev(data[st:st+min_len])

        elif mode == 'max': # max 
            if max_len == data_len:
                st = 0
            else:
                st = randstat.randint(max_len - data_len)
            data = activelev(data)

            if st > 0:
                data_t = np.zeros(shape)
                data_t[-st:-st+data_len] = data 

        elif mode == 'first':
            if max_len > data_len:
                st = randstat.randint(max_len - data_len)
                data_t = np.zeros(shape)
                data_t[st:st+data_len] = data
                data = data_t
            elif max_len < data_len:
                st = randstat.randint(data_len - max_len)
                data = data[st:st+max_len]
            data = activelev(data)
        
        weight = 10**(snr/40)
        data = data * weight
        '''
        if add_white is True:
            white = randstat.randn(data.shape[0])
            white = np.clip(white, -1, 1)
            max_t = np.max(np.abs(data)) 
            white = max_t*0.003*white
            data += white
            add_white = False
        '''
        
        idx += 1
        mix += data
        processed_wavs.append(data)
    mix_fac = 1./np.max(np.abs([mix]+processed_wavs))*scale 
    mix *= mix_fac
    processed_wavs = [ x*mix_fac for x in processed_wavs]
    return mix, processed_wavs, mix_fac

def test_activelev(path):
    data, fs = sf.read(path)
    data = activelev(data)
    sf.write(path[:-4]+'_0db.wav', data, fs)

def test_Mixer():
    speech_scp = '../data/train_daps.lst'
    bk_scp = '../data/train_noisy.lst'
    r = np.random.RandomState(30)
    mixer = Mixer(mix_scps=[speech_scp, bk_scp], load_memory=False,mono=True, sample_rate=44100) 

    for idx in range(1000):
        a=mixer(None, start=44100*10,segement_length=44100*5, randstat=r)
        print(a[0].shape)
        sf.write('{:d}_1.wav'.format(idx),a[0],44100) 
        sf.write('{:d}_2.wav'.format(idx),np.array(a[1]).T,44100) 
        #sf.write('{:d}_3.wav'.format(idx),a[2],44100) 
        #print(a[2:])
        break

def test_RIR():
    speech_scp = '../../debug/dns_clean.lst'
    bk_scp = '../../debug/dns_noise.lst'
    rir_scp = '../../debug/rir.lst'
    r = np.random.RandomState(30)
    mixer = Mixer(mix_scps=[speech_scp, bk_scp], rir_scps=rir_scp, load_memory=False,mono=True, sample_rate=16000) 

    for idx in range(1000):
        a=mixer(None,start=0,segement_length=16000*5, randstat=r)
        print(a[0].shape)
        if a[2] is not None :
            sf.write('{:d}_1.wav'.format(idx),a[0],16000) 
            sf.write('{:d}_2.wav'.format(idx),np.array(a[1]).T,16000) 
            sf.write('{:d}_3.wav'.format(idx),np.array(a[2]).T,16000) 
            break

def fuck():
    get_wave_header('./0_1.wav',16000)

if __name__ == "__main__":
    #test_activelev('./E10051.wav')
    #test_Mixer()
    #test_RIR()
    fuck()
    input('end')
