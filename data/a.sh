#!/bin/bash

nums=3000
shuf ./dns_clean.lst > s 
head -n ${nums} s > cv_clean.lst
tail -n +${nums} s > tr_clean.lst


shuf ./dns_noise.lst > s 
head -n 10000 s > cv_noise.lst
tail -n +10000 s > tr_noise.lst


shuf ./rir.lst > s 
head -n 10000 s > cv_rir.lst
tail -n +10000 s > tr_rir.lst
