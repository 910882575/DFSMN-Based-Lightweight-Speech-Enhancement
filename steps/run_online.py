
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import argparse
import torch.nn.parallel.data_parallel as data_parallel
import numpy as np
import scipy
import scipy.io as sio
import torch.optim as optim
import time
import multiprocessing
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(
    os.path.dirname(sys.path[0]) + '/tools/speech_processing_toolbox')

from model.model import Net as Model 

from tools.misc import get_learning_rate, save_checkpoint, reload_for_eval, reload_model, setup_lr
from tools.time_dataset import make_loader, Processer, DataReader
from tools.online_dataset import make_dataloader

import soundfile as sf
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

def train(model, args, device, writer):
    print('preparing data...')
    dataloader = make_dataloader(
        args.tr_clean_list,
        args.tr_noise_list,
        args.tr_rir_list,
        batch_size=args.batch_size,
        repeate=1,
        segement_length=8,
        sample_rate=args.sample_rate,
        num_workers=args.num_threads,
            )

    print_freq = 100
    num_batch = len(dataloader)
    params = model.get_params(args.weight_decay)
    optimizer = optim.Adam(params, lr=args.learn_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=1, verbose=True)
    
    if args.retrain:
        start_epoch, step = reload_model(model, optimizer, args.exp_dir,
                                         args.use_cuda)
    else:
        start_epoch, step = 0, 0
    print('---------PRERUN-----------')
    lr = get_learning_rate(optimizer)
    print('(Initialization)')
    val_loss, val_sisnr = validation(model, args, lr, -1, device)
    writer.add_scalar('Loss/Train', val_loss, step)
    writer.add_scalar('Loss/Cross-Validation', val_loss, step)
    
    writer.add_scalar('SISNR/Train', -val_sisnr, step)
    writer.add_scalar('SISNR/Cross-Validation', -val_sisnr, step)
    
    warmup_epoch = 6
    warmup_lr = args.learn_rate/(4*warmup_epoch)

    for epoch in range(start_epoch, args.max_epoch):
        torch.manual_seed(args.seed + epoch)
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed + epoch)
        model.train()
        loss_total = 0.0
        loss_print = 0.0 

        sisnr_total = 0.0 
        sisnr_print = 0.0  
        '''
        if epoch == 0 and warmup_epoch > 0:
            print('Use warmup stragery, and the lr is set to {:.5f}'.format(warmup_lr))
            setup_lr(optimizer, warmup_lr)
            warmup_lr *= 4*(epoch+1)
        elif epoch == warmup_epoch:
            print('The warmup was end, and the lr is set to {:.5f}'.format(args.learn_rate))
            setup_lr(optimizer, args.learn_rate)
        '''


     
        stime = time.time()
        lr = get_learning_rate(optimizer)
        for idx, data in enumerate(dataloader):
            torch.cuda.empty_cache()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            model.zero_grad()
            est_spec, est_wav = data_parallel(model, (inputs,))
            '''
            if epoch > 8:
                gth_spec, gth_wav = data_parallel(model, (labels,))
            else:
                gth_spec = data_parallel(model.stft, (labels))[0]
            '''  
            #gth_spec = data_parallel(model.stft, (labels))
            #loss = model.loss(est_spec, gth_spec, loss_mode='MSE')
            #loss.backward()
            sisnr = model.loss(est_wav, labels, loss_mode='SI-SNR')
            sisnr.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            
            step += 1

            #loss_total += loss.data.cpu()
            #loss_print += loss.data.cpu()

            sisnr_total += sisnr.data.cpu()
            sisnr_print += sisnr.data.cpu()

            loss_total = sisnr_total
            loss_print = sisnr_print
            del est_wav, est_spec
            if (idx+1) % 3000 == 0:
                save_checkpoint(model, optimizer, -1, step, args.exp_dir)
            if (idx + 1) % print_freq == 0:
                eplashed = time.time() - stime
                speed_avg = eplashed / (idx+1)
                loss_print_avg = loss_print / print_freq
                sisnr_print_avg = sisnr_print / print_freq
                print('Epoch {:3d}/{:3d} | batches {:5d}/{:5d} | lr {:1.4e} |'
                      '{:2.3f}s/batches | loss {:2.6f} |'
                      'SI-SNR {:2.4f} '.format(
                          epoch, args.max_epoch, idx + 1, num_batch, lr,
                          speed_avg, 
                          loss_print_avg,
                          -sisnr_print_avg,
                          
                          ))
                sys.stdout.flush()
                writer.add_scalar('Loss/Train', loss_print_avg, step)
                writer.add_scalar('SISNR/Train', -sisnr_print_avg, step)
                loss_print = 0.0
                sisnr_print=0.0
        eplashed = time.time() - stime
        loss_total_avg = loss_total / num_batch
        sisnr_total_avg = sisnr_total / num_batch
        print(
            'Training AVG.LOSS |'
            ' Epoch {:3d}/{:3d} | lr {:1.4e} |'
            ' {:2.3f}s/batch | time {:3.2f}mins |'
            ' loss {:2.6f} |'
            ' SISNR {:2.4f}|'
                    .format(
                                    epoch + 1,
                                    args.max_epoch,
                                    lr,
                                    eplashed/num_batch,
                                    eplashed/60.0,
                                    loss_total_avg.item(),
                                    -sisnr_total_avg.item()
                        ))
        val_loss, val_sisnr= validation(model, args, lr, epoch, device)
        writer.add_scalar('Loss/Cross-Validation', val_loss, step)
        writer.add_scalar('SISNR/Cross-Validation', -val_sisnr, step)
        writer.add_scalar('learn_rate', lr, step) 
        if val_loss > scheduler.best:
            print('Rejected !!! The best is {:2.6f}'.format(scheduler.best))
        else:
            save_checkpoint(model, optimizer, epoch + 1, step, args.exp_dir, mode='best_model')
        scheduler.step(val_loss)
        sys.stdout.flush()
        stime = time.time()


def validation(model, args, lr, epoch, device):
    '''
    dataloader, dataset = make_loader(
        args.cv_list,
        args.batch_size,
        8,
        num_workers=args.num_threads,
        processer=Processer(
            ))
    '''
    dataloader = make_dataloader(
        args.cv_clean_list,
        args.cv_noise_list,
        args.cv_rir_list,
        batch_size=args.batch_size,
        segement_length=8,
        repeate=1,
        sample_rate=args.sample_rate,
        num_workers=args.num_threads,
      )
    model.eval()
    loss_total = 0.0 
    sisnr_total = 0.0 
    num_batch = len(dataloader)
    stime = time.time()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            est_spec, est_wav = data_parallel(model, (inputs,))
            #gth_spec = data_parallel(model.stft, (labels))
            #loss = model.loss(est_spec, gth_spec, loss_mode='MSE')
            sisnr = model.loss(est_wav, labels, loss_mode='SI-SNR')
            loss = sisnr
            loss_total += loss.data.cpu()
            sisnr_total += sisnr.data.cpu()

        etime = time.time()
        eplashed = (etime - stime) / num_batch
        loss_total_avg = loss_total / num_batch
        sisnr_total_avg = sisnr_total / num_batch

    print('CROSSVAL AVG.LOSS | Epoch {:3d}/{:3d} '
          '| lr {:.4e} | {:2.3f}s/batch| time {:2.1f}mins '
          '| loss {:2.6f} |'
          '| SISNR {:2.4f} '.format(
                        epoch + 1,
                        args.max_epoch,
                        lr,
                        eplashed,
                        (etime - stime)/60.0,
                        loss_total_avg,
                        -sisnr_total_avg,
              ))
    sys.stdout.flush()
    return loss_total_avg, sisnr_total_avg


def decode(model, args, device):
    model.eval()
    with torch.no_grad():
        
        data_reader = DataReader(
            args.tt_list,
            sample_rate=args.sample_rate)
        output_wave_dir = os.path.join(args.exp_dir, 'rec_wav/')
        if not os.path.isdir(output_wave_dir):
            os.mkdir(output_wave_dir)
        num_samples = len(data_reader)
        print('Decoding...')
        for idx in tqdm(range(num_samples)):
            inputs, utt_id, nsamples = data_reader[idx]
            inputs = torch.from_numpy(inputs)
            inputs = inputs.to(device)
            window = int(args.sample_rate*4)
            decode_do_segement=False
            b,t = inputs.size()
            if t > int(1.5*window) and decode_do_segement:
                outputs = np.zeros(t)
                stride = int(window*0.75)
                give_up_length=(window - stride)//2
                current_idx = 0
                while current_idx + window < t:
                    tmp_input = inputs[:,current_idx:current_idx+window]
                    tmp_output = model(tmp_input,)[1][0].cpu().numpy()
                    if current_idx == 0:
                        outputs[current_idx:current_idx+window-give_up_length] = tmp_output[:-give_up_length]

                    else:
                        outputs[current_idx+give_up_length:current_idx+window-give_up_length] = tmp_output[give_up_length:-give_up_length]
                    current_idx += stride 
                if current_idx < t:
                    tmp_input = inputs[:,current_idx:current_idx+window]
                    tmp_output = model(tmp_input)[1][0].cpu().numpy()
                    length = tmp_output.shape[0]
                    outputs[current_idx+give_up_length:current_idx+length] = tmp_output[give_up_length:]
            else:
                outputs = model(inputs)[1][0].cpu().numpy()
            if outputs.shape[0] > nsamples:
                outputs = outputs[:nsamples]
            else:
                outputs = np.pad(outputs,[0,nsamples-outputs.shape[0]]) 

            outputs[:800] =0
            outputs[-800:] =0
            # this just for plot mask 
            #amp, mask, phase = model(inputs)[2] 
            #np.save(utt_id, [amp.cpu().numpy(), mask.cpu().numpy(), phase.cpu().numpy()]) 
            sf.write(os.path.join(output_wave_dir, utt_id), outputs, args.sample_rate) 
            sf.write(os.path.join(output_wave_dir, utt_id), outputs, args.sample_rate) 
        print('Decode Done!!!')


def main(args):
    device = torch.device('cuda' if args.use_cuda else 'cpu')
    args.sample_rate = {
        '8k':8000,
        '16k':16000,
        '24k':24000,
        '48k':48000,
    }[args.sample_rate]
    model = Model(
        rnn_layers=args.rnn_layers,
        rnn_units=args.rnn_units,
        win_len=args.win_len,
        win_inc=args.win_inc,
        fft_len=args.fft_len,
        win_type=args.win_type,
        mode=args.target_mode,
    )
    if not args.log_dir:
        writer = SummaryWriter(os.path.join(args.exp_dir, 'tensorboard'))
    else:
        writer = SummaryWriter(args.log_dir)
    model.to(device)
    if not args.decode:
        train(model, FLAGS, device, writer)
    reload_for_eval(model, FLAGS.exp_dir, FLAGS.use_cuda)
    decode(model, args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('PyTorch Version ENhancement')
    parser.add_argument('--decode', type=int, default=0, help='if decode')
    parser.add_argument(
        '--exp-dir',
        dest='exp_dir',
        type=str,
        default='exp/cldnn',
        help='the exp dir')
    parser.add_argument(
        '--tr-noise-list', dest='tr_noise_list', type=str, help='the train data list')
    parser.add_argument(
        '--tr-clean-list', dest='tr_clean_list', type=str, help='the train data list')
    parser.add_argument(
        '--tr-rir-list', dest='tr_rir_list', type=str, help='the train data list')
    
    parser.add_argument(
        '--cv-noise-list', dest='cv_noise_list', type=str, help='the dev data list')
    parser.add_argument(
        '--cv-clean-list', dest='cv_clean_list', type=str, help='the dev data list')
    parser.add_argument(
        '--cv-rir-list',  dest='cv_rir_list', type=str, help='the dev data list')
    

    parser.add_argument(
        '--tt-list', dest='tt_list', type=str, help='the test data list')
    parser.add_argument(
        '--rnn-layers',
        dest='rnn_layers',
        type=int,
        default=2,
        help='the num hidden rnn layers')
    parser.add_argument(
        '--rnn-units',
        dest='rnn_units',
        type=int,
        default=512,
        help='the num hidden rnn units')
    parser.add_argument(
        '--learn-rate',
        dest='learn_rate',
        type=float,
        default=0.001,
        help='the learning rate in training')
    parser.add_argument(
        '--max-epoch',
        dest='max_epoch',
        type=int,
        default=20,
        help='the max epochs')

    parser.add_argument(
        '--dropout',
        dest='dropout',
        type=float,
        default=0.2,
        help='the probility of dropout')
    parser.add_argument(
        '--left-context',
        dest='left_context',
        type=int,
        default=1,
        help='the left context to add')
    parser.add_argument(
        '--right-context',
        dest='right_context',
        type=int,
        default=1,
        help='the right context to add')
    parser.add_argument(
        '--input-dim',
        dest='input_dim',
        type=int,
        default=257,
        help='the input dim')
    parser.add_argument(
        '--output-dim',
        dest='output_dim',
        type=int,
        default=257,
        help='the output dim')
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        help='the batch size in train')
    parser.add_argument(
        '--use-cuda', dest='use_cuda', default=1, type=int, help='use cuda')
    parser.add_argument(
        '--seed', dest='seed', type=int, default=20, help='the random seed')
    parser.add_argument(
        '--log-dir',
        dest='log_dir',
        type=str,
        default=None,
        help='the random seed')
    parser.add_argument(
        '--num-threads', dest='num_threads', type=int, default=10)
    parser.add_argument(
        '--window-len',
        dest='win_len',
        type=int,
        default=400,
        help='the window-len in enframe')
    parser.add_argument(
        '--window-inc',
        dest='win_inc',
        type=int,
        default=100,
        help='the window include in enframe')
    parser.add_argument(
        '--fft-len',
        dest='fft_len',
        type=int,
        default=512,
        help='the fft length when in extract feature')
    parser.add_argument(
        '--window-type',
        dest='win_type',
        type=str,
        default='hamming',
        help='the window type in enframe, include hamming and None')
    parser.add_argument(
        '--kz-freq',
        dest='kz_freq',
        type=int,
        default=3,
        help='the kernel_size in freq dim')
    parser.add_argument(
        '--kz-time',
        dest='kz_time',
        type=int,
        default=1,
        help='the kernel_size in time dim')
    parser.add_argument(
        '--num-gpu',
        dest='num_gpu',
        type=int,
        default=1,
        help='the num gpus to use')
    parser.add_argument(
        '--target-mode',
        dest='target_mode',
        type=str,
        default='MSA',
        help='the type of target, MSA, PSA, PSM, IBM, IRM...')
    
    parser.add_argument(
        '--weight-decay', dest='weight_decay', type=float, default=0.00001)
    parser.add_argument(
        '--clip-grad-norm', dest='clip_grad_norm', type=float, default=5.)
    parser.add_argument(
        '--sample-rate', dest='sample_rate', type=str, default='16k')
    parser.add_argument('--retrain', dest='retrain', type=int, default=0)
    FLAGS, _ = parser.parse_known_args()
    FLAGS.use_cuda = FLAGS.use_cuda and torch.cuda.is_available()
    os.makedirs(FLAGS.exp_dir, exist_ok=True)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    if FLAGS.use_cuda:
        torch.cuda.manual_seed(FLAGS.seed)
    import pprint
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__dict__)
    print(FLAGS.win_type)
    main(FLAGS)
