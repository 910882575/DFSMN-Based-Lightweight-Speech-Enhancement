#!/bin/bash 
input_dim=257
output_dim=257
left_context=0
right_context=0
lr=0.001



win_len=400
win_inc=100
fft_len=512

sample_rate=16k
win_type=hanning
batch_size=32
max_epoch=50
rnn_units=200
rnn_layers=2
tt_list=data/tt.lst
tr_list=data/tr.lst
cv_list=data/cv.lst


cv_list='data/dev.scp'
tr_list='data/train.scp'
tt_list='data/t'
tt_list='data/test.scp'

dropout=0.0
kernel_size=6
kernel_num=9
nropout=0.2
retrain=0
sample_rate=16k
num_gpu=1
batch_size=$[num_gpu*batch_size]
target_mode=TMS
target_mode=MSA

tr_noise_list='./data/tr_noise.lst'
tr_clean_list='./data/tr_clean.lst'
tr_rir_list='./data/tr_rir.lst'


cv_noise_list='./data/cv_noise.lst'
cv_clean_list='./data/cv_clean.lst'
cv_rir_list='./data/cv_rir.lst'


save_name=clp_wolog_df_real2complex_res_10layer_Semi_2w_MSA_DFSMN-fixbug_Time-SiSNR_dns_${target_mode}_${lr}_${rnn_layers}_${rnn_units}_${sample_rate}_${win_len}_${win_inc}

exp_dir=exp/${save_name}
if [ ! -d ${exp_dir} ] ; then
    mkdir -p ${exp_dir}
fi

stage=1

if [ $stage -le 1 ] ; then
    #CUDA_VISIBLE_DEVICES='6' nohup python -u ./steps/run.py \
/home/work_nfs/common/tools/pyqueue_asr.pl \
    -q g.q --gpu 1 --num-threads ${num_gpu} \
    ${exp_dir}/${save_name}.log \
    python -u ./steps/run_online.py \
    --decode=0 \
    --fft-len=${fft_len} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
    --window-len=${win_len} \
    --window-inc=${win_inc} \
    --exp-dir=${exp_dir} \
    --tr-noise-list=${tr_noise_list} \
    --tr-clean-list=${tr_clean_list} \
    --tr-rir-list=${tr_rir_list} \
    --cv-noise-list=${cv_noise_list} \
    --cv-clean-list=${cv_clean_list} \
    --cv-rir-list=${cv_rir_list} \
    --tt-list=${tt_list} \
    --retrain=${retrain} \
    --rnn-layers=${rnn_layers} \
    --rnn-units=${rnn_units} \
    --learn-rate=${lr} \
    --max-epoch=${max_epoch} \
    --dropout=${dropout} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
    --left-context=${left_context} \
    --right-context=${right_context} \
    --batch-size=${batch_size} \
    --kernel-size=${kernel_size} \
    --kernel-num=${kernel_num} \
    --sample-rate=${sample_rate} \
    --target-mode=${target_mode} \
    --window-type=${win_type} &
    exit 0
fi

if [ $stage -le 2 ] ; then 
    CUDA_VISIBLE_DEVICES='2' python -u ./steps/run.py \
    --decode=1 \
    --fft-len=${fft_len} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
    --window-len=${win_len} \
    --window-inc=${win_inc} \
    --exp-dir=${exp_dir} \
    --tr-list=${tr_list} \
    --cv-list=${cv_list} \
    --tt-list=${tt_list} \
    --retrain=${retrain} \
    --rnn-layers=${rnn_layers} \
    --rnn-units=${rnn_units} \
    --learn-rate=${lr} \
    --max-epoch=${max_epoch} \
    --dropout=${dropout} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
    --left-context=${left_context} \
    --right-context=${right_context} \
    --batch-size=${batch_size} \
    --kernel-size=${kernel_size} \
    --kernel-num=${kernel_num} \
    --sample-rate=${sample_rate} \
    --target-mode=${target_mode} \
    --window-type=${win_type} 
    #exit 0
fi

if [ $stage -le 3 ] ; then

#for snr in -5 0 5 10 ; do 
#for snr in 15 20 ; do 
for snr in -5 0 5 10 15 20 ; do 
    #dataset_name=aishell
    dataset_name=wsj0_musan_1k
    tgt=semi_rnn_${target_mode}_${dataset_name}_${snr}db
    #clean_wav_path="/search/odin/huyanxin/workspace/se-cldnn-torch/data/wavs/test_clean_${snr}/"
    #noisy_wav_path="/search/odin/huyanxin/workspace/se-cldnn-torch/data/wavs/test_noisy_${snr}/"
    #clean_wav_path="/search/odin/huyanxin/workspace/se-resnet/data/wavs/test_wsj0_clean_${snr}/"
    #noisy_wav_path="/search/odin/huyanxin/workspace/se-resnet/data/wavs/test_wsj0_noisy_${snr}/"
    #clean_wav_path="/search/odin/huyanxin/workspace/se-cldnn-torch/data/wavs/test_${dataset_name}_clean_${snr}/"
    #noisy_wav_path="/search/odin/huyanxin/workspace/se-cldnn-torch/data/wavs/test_${dataset_name}_noisy_${snr}/"
    
    clean_wav_path="/dockerdata/yanxinhu_data/wavs/test_wsj0_dcase_clean_${snr}/"
    noisy_wav_path="/dockerdata/yanxinhu_data/wavs/test_wsj0_dcase_noisy_${snr}/"
    
    clean_wav_path="/dockerdata/yanxinhu_data/test1k/wavs/test_wsj0_musan_clean_${snr}/"
    noisy_wav_path="/dockerdata/yanxinhu_data/test1k/wavs/test_wsj0_musan_noisy_${snr}/"

    enh_wav_path=${exp_dir}/test_${dataset_name}_noisy_${snr}/
    enh_wav_path=${exp_dir}/rec_wav/
    find ${noisy_wav_path} -iname "*.wav" > wav.lst
#    CUDA_VISIBLE_DEVICES='2' python -u ./steps/run_rnn.py \
#    --decode=1 \
#    --fft-len=${fft_len} \
#    --input-dim=${input_dim} \
#    --output-dim=${output_dim} \
#   --window-len=${win_len} \
#    --window-inc=${win_inc} \
#    --exp-dir=${exp_dir} \
#    --tr-noise=${tr_noise_list} \
#    --tr-clean=${tr_clean_list} \
#    --cv-noise=${cv_noise_list} \
#    --cv-clean=${cv_clean_list} \
#    --tt-list=wav.lst \
#    --retrain=${retrain} \
#    --rnn-layers=${rnn_layers} \
#    --rnn-units=${rnn_units} \
#    --learn-rate=${lr} \
#    --max-epoch=${max_epoch} \
#    --dropout=${dropout} \
#    --input-dim=${input_dim} \
#    --output-dim=${output_dim} \
#    --left-context=${left_context} \
#    --right-context=${right_context} \
#    --target-mode=${target_mode} \
#    --batch-size=${batch_size} \
#    --kernel-size=${kernel_size} \
#    --sample-rate=${sample_rate} \
#    --window-type=${win_type}  || exit 1 # > ${exp_dir}/train.log &
#    mv ${exp_dir}/rec_wav ${enh_wav_path}
    
    ls $noisy_wav_path > t
    python ./tools/eval_objective.py --wav_list=t --result_list=${tgt}.csv --pathe=${enh_wav_path}\
    --pathc=${clean_wav_path} --pathn=${noisy_wav_path} ||exit 1
done

fi
