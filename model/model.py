import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]) + '/model')
from show import show_params, show_model
import torch.nn.functional as F
from conv_stft import ConvSTFT, ConviSTFT
from loss import si_snr, sd_snr
from dfsmn import  DFSMN
from deepfilter import DeepFilter

class ComplexLinearProjection(nn.Module):
    
    def __init__(self, in_dim):
        super(ComplexLinearProjection, self).__init__()
        
        self.input_dim = in_dim

        self.real = nn.Conv1d(self.input_dim, self.input_dim, 1, bias=False)
        self.imag = nn.Conv1d(self.input_dim, self.input_dim, 1, bias=False)
    
    def forward(self,real,imag):
        inputs = torch.cat([real,imag],0) 
        ri2r = self.real(inputs)
        ri2i = self.imag(inputs)
        
        r2r, i2r = torch.chunk(ri2r, 2, 0)
        r2i, i2i = torch.chunk(ri2i, 2, 0)
        real = r2r - i2i
        imag = r2i + i2r
        
        outputs = torch.sqrt(real**2+imag**2+1e-8)
        
        return outputs

class Net(nn.Module):

    def __init__(self, rnn_layers=2, rnn_units=128, win_len=400, win_inc=100, fft_len=512, win_type='hanning', mode='TCS'):
        super(Net, self).__init__()

        # for fft 
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type 

        input_dim = win_len
        output_dim = win_len
        
        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = rnn_layers
        fix=True
        self.fix = fix
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        
        in_dim = self.fft_len//2
        out_dim = self.fft_len//2
        
        self.clp = ComplexLinearProjection(in_dim)
        self.bn = nn.BatchNorm1d(in_dim, affine=False)

        self.fsmn1 = DFSMN(in_dim, rnn_units, out_dim, 3,3,1,1)
        self.fsmn2 = DFSMN(in_dim, rnn_units, out_dim, 7,3,1,1)
        self.fsmn3 = DFSMN(in_dim, rnn_units, out_dim, 7,3,1,1)
        self.fsmn4 = DFSMN(in_dim, rnn_units, out_dim, 7,3,1,1)
        self.fsmn5 = DFSMN(in_dim, rnn_units, out_dim, 7,3,1,1)
        self.fsmn6 = DFSMN(in_dim, rnn_units, out_dim, 7,3,1,1)
        self.fsmn7 = DFSMN(in_dim, rnn_units, out_dim, 7,3,0,1)
        self.fsmn8 = DFSMN(in_dim, rnn_units, out_dim, 7,3,0,1)
        self.fsmn9 = DFSMN(in_dim, rnn_units, out_dim, 7,3,0,1)
        self.fsmn10 = DFSMN(in_dim, rnn_units, out_dim*2, 3,3,0,1)

        self.df = DeepFilter(1,2)
        show_params(self)


    def forward(self, inputs, lens=None):
        
        complex_specs = self.stft(inputs)
        
        r_specs, i_specs = torch.chunk(complex_specs,2,1)
        specs = self.clp(r_specs[:,1:],i_specs[:,1:])#torch.sqrt(r_specs**2+i_specs**2)

        specs = self.bn(specs)

        out, outh = self.fsmn1(specs)
        out = torch.relu(out)
        out, outh = self.fsmn2(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn3(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn4(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn5(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn6(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn7(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn8(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn9(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn10(out,outh)
        #mask = torch.sigmoid(out)
        
        r_mask, i_mask = torch.chunk(out,2,1)
        ''' 
        phase = torch.atan2(i_mask, r_mask+1e-8)
        mag = torch.sqrt(r_mask**2+i_mask**2+1e-8) 
        mag = torch.clamp(mag,0,10) 
        r_mask = mag*torch.cos(phase)
        i_mask = mag*torch.sin(phase) # conj
        
        
        out_spec = self.df([r_specs,i_specs], [r_mask, i_mask])
        ''' 
        r_mask = F.pad(r_mask,[0,0,1,0])
        i_mask = F.pad(i_mask,[0,0,1,0])
        r_out_spec = r_mask*r_specs - i_mask*i_specs
        i_out_spec = r_mask*i_specs + i_mask*r_specs
        out_spec = torch.cat([r_out_spec,i_out_spec],1) 
        
        out_wav = self.istft(out_spec)
         
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav,-1,1)
        return out_spec,  out_wav

    def get_params(self, weight_decay=0.0):
            # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        return params

    def loss(self, inputs, labels, loss_mode='SI-SNR'):
       
        if loss_mode == 'MSE':
            b,t,d = inputs.shape 
            #gth_spec = self.stft(labels)
            return F.mse_loss(inputs, labels, reduction='mean')*d

        elif loss_mode == 'SI-SNR':
            #return -torch.mean(si_snr(inputs, labels))
            return -(si_snr(inputs, labels))
        elif loss_mode == 'SD-SNR':
            #return -torch.mean(si_snr(inputs, labels))
            return -(sd_snr(inputs, labels))
        elif loss_mode == 'MAE':
            gth_spec, gth_phase = self.stft(labels) 
            b,d,t = inputs.shape 
            return torch.mean(torch.abs(inputs-gth_spec))*d


if __name__ == '__main__':
    torch.manual_seed(10)
    inputs = torch.randn([10,16000*4]).clamp_(-1,1)
    labels = torch.randn([10,16000*4]).clamp_(-1,1)
    inputs = inputs #labels
    net = Net()
    outputs = net(inputs)[1]
    loss = net.loss(outputs, labels, loss_mode='SI-SNR')
    print(loss)

