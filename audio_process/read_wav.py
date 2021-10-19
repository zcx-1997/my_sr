# -*- coding: utf-8 -*-

"""
@Time : 2021/10/19
@Author : Lenovo
@File : read_wav
@Description : 
"""
import time
import torchaudio
import wave
import librosa
from scipy.io import wavfile

import numpy as np
import matplotlib.pyplot as plt

def load_wav_torchaudio(wav_path):

    print("===1.torchaudio=======================")
    start_t = time.time()
    audio, sr = torchaudio.load(wav_path)  # 16000 torch.Size([1, 129921])
    print("cost time:", time.time()-start_t)
    print(sr, audio.shape)
    print(audio.std(1))
    print("mean:{:.10f}\nstd:{}\nmax :{}\nmin :{}".format(audio.mean(1).item(), audio.std(1).item(),audio.max(1)[0].item(),audio.min(1)[0].item()))
    '''
    mean:-0.0000014744
    max :0.494537353515625
    min :-0.377227783203125
    '''
    t = np.arange(audio.shape[1])
    plt.plot(t, audio[0])
    plt.show()
    return

def load_wav_librosa(wav_path):

    print("===2.librosa=======================")
    start_t = time.time()
    audio, sr = librosa.load(wav_path,sr=16000)  # 16000 (129921,)
    print("cost time:", time.time() - start_t)
    print(sr, audio.shape)
    print("mean:{:.10f}\nmax :{}\nmin :{}".format(audio.mean(), audio.max(), audio.min()))
    '''
    mean: -0.0000014744
    max: 0.494537353515625
    min: -0.377227783203125
    '''
    # t = np.arange(len(audio))
    # plt.plot(t, audio)
    # plt.show()

    return

def load_wav_scipyiowavfile(wav_path):

    print("===3.scipy.io.wavfile=======================")
    start_t = time.time()
    sr, audio = wavfile.read(wav_path)  # 16000 (129921,)
    # print(audio[0])  # 2302  （10）2302 = （16）8fe
    print("cost time:", time.time() - start_t)
    print(sr, audio.shape)
    print("mean:{:.10f}\nmax :{}\nmin :{}".format(audio.mean(), audio.max(), audio.min()))
    '''
    mean:-0.0483139754
    max :16205
    min :-12361
    '''
    # t = np.arange(len(audio))
    # plt.plot(t, audio)
    # plt.show()

    return

def load_wav_wave(wav_path):

    print("===4.wave=======================")
    start_t = time.time()
    wav = wave.open(wav_path,"rb") # 打开一个wav格式的声音文件流
    params = wav.getparams()
    # _wave_params(nchannels=1, sampwidth=2, framerate=16000, nframes=129921, comptype='NONE', compname='not compressed')

    num_frame = wav.getnframes()             # 获取帧数     129921
    num_channel = wav.getnchannels()         # 获取声道数    1
    framerate = wav.getframerate()           # 获取帧速率    16000
    num_sample_width = wav.getsampwidth()    # 获取实例的比特宽度，即每一帧的字节数 2

    str_data = wav.readframes(num_frame)  # 读取全部的帧 返回的值是二进制数据 len=259842
    print(str_data[0:2])  # b'\xfe\x08' （10）2302 = （16）8fe
    wav.close()  # 关闭流

    wave_data = np.frombuffer(str_data, dtype=np.short)  # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵  (129921, 1)
    audio = wave_data.T  # 将矩阵转置 (1, 129921)
    print("cost time:", time.time() - start_t)

    # 归一化
    audio = 0.08 * (audio-np.mean(audio)) / (np.std(audio)+1e-5)

    print("mean:{:.10f}\nstd:{}\nmax :{}\nmin :{}".format(audio.mean(1).item(),audio.std(1).item(),audio.max(1)[0].item(),audio.min(1)[0].item()))
    '''
    mean:-0.0483139754
    max :16205
    min :-12361
    '''
    # t = np.arange(audio.shape[1])
    # plt.plot(t, audio[0])
    # plt.show()
    return


if __name__ == '__main__':
    wav_path = r'00001.wav'
    load_wav_torchaudio(wav_path)
    # load_wav_librosa(wav_path)
    # load_wav_scipyiowavfile(wav_path)
    # load_wav_wave(wav_path)