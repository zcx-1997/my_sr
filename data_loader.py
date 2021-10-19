# -*- coding: utf-8 -*-

"""
@Time : 2021/10/19
@Author : Lenovo
@File : data_loader
@Description : 
"""

import os
import random
import time
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from python_speech_features import fbank, mfcc, delta

from vad_segment import vad_filter
from utils import Timer

from hparam import hparam as hp

class Vox1_TrainDataset(Dataset):
    def __init__(self):
        self.random = random
        self.train_txt = hp.data.train_txt
        with open(self.train_txt, 'r') as f:
            self.wavs_labels = f.readlines()
        self.random.shuffle(self.wavs_labels)
        self.fixed_time = hp.data.fixed_time

    def __getitem__(self, idx):
        wav_label = self.wavs_labels[idx].split()
        wav_path = wav_label[0]
        label = int(wav_label[1])
        feats = self._load_data(wav_path)
        feats = torch.tensor(feats, dtype=torch.float32)
        label = torch.tensor(label)
        return feats, label

    def __len__(self):
        return len(self.wavs_labels)

    def _load_data(self, wav_path):
        audio, sr = vad_filter(wav_path)
        while (len(audio)/sr) < self.fixed_time:
            audio = np.append(audio,audio)
        audio_f = audio[:sr * self.fixed_time]
        mfcc_f = mfcc(audio_f, sr,numcep=hp.data.nmels, appendEnergy=True)
        feats = mfcc_f
        return feats



class Vox1_TestDataset(Dataset):
    def __init__(self):
        self.random = random
        self.test_dir = r'/home/ubuntu/datasets/vox1/vox1_test_wav/wav'
        self.test_txt = r'/home/ubuntu/datasets/vox1/vox1_txt/veri_test.txt'
        with open(self.test_txt, 'r') as f:
            self.test_pairs = f.readlines()
        self.random.shuffle(self.test_pairs)
        self.fixed_time = hp.data.fixed_time

    def __getitem__(self, idx):
        label_pairs = self.test_pairs[idx].split()
        label = int(label_pairs[0])
        enroll_path = os.path.join(self.test_dir,label_pairs[1])
        test_path = os.path.join(self.test_dir,label_pairs[2])

        enroll_feats = self._load_data(enroll_path)
        enroll_feats = torch.tensor(enroll_feats, dtype=torch.float32)
        test_feats = self._load_data(test_path)
        test_feats = torch.tensor(test_feats, dtype=torch.float32)
        label = torch.tensor(label)
        return enroll_feats, test_feats, label


    def __len__(self):
        return len(self.test_pairs)

    def _load_data(self, wav_path):
        audio, sr = vad_filter(wav_path)
        while (len(audio) / sr) < self.fixed_time:
            audio = np.concatenate((audio, audio))
            # audio = torch.cat((audio, audio))
        audio_f = audio[:sr * self.fixed_time]
        mfcc_f = mfcc(audio_f, sr, numcep=hp.data.nmels, appendEnergy=True)
        feats = mfcc_f
        return feats


if __name__ == '__main__':
    timer = Timer()
    train_db = Vox1_TrainDataset()
    train_loader = DataLoader(train_db,batch_size=64,num_workers=6,pin_memory=True,shuffle=False,drop_last=True)
    print(len(train_db))
    print(len(train_loader))

    # timer.start()
    # x, y = next(iter(train_loader))
    # print('time cost=', timer.stop())

    # for i, (x,y) in enumerate(train_loader):
    #     print(i,time.ctime())

