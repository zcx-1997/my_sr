# -*- coding: utf-8 -*-

"""
@Time : 2021/10/19
@Author : Lenovo
@File : create_trials_txt
@Description : 
"""
import os
import glob
import random

from hparam import hparam as hp

train_txt = r'data_txt/vox1_train_500.txt'

train_dir = os.path.join(hp.data.dataset_dir, hp.data.train_root)
spks_list = os.listdir(train_dir)
random.shuffle(spks_list)
# spks_list = spks_list[:100]

# 创建trian_txt_all
with open(train_txt, 'w') as f:
    for spk_name in spks_list:
        spk_id = int(spk_name[-4:])
        if spk_id < 500:
            wavs_list = glob.glob(os.path.join(train_dir, spk_name, '*/*'))
            for wav_path in wavs_list:
                # print(wav_path, spk_id)
                f.write(wav_path+' '+str(spk_id)+'\n')