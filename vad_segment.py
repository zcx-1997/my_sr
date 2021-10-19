# -*- coding: utf-8 -*-

"""
@Time : 2021/10/19
@Author : Lenovo
@File : vad_segment
@Description : 
"""

import wave
import librosa
import collections
import contextlib
import numpy as np
import torch

from matplotlib import pyplot as plt

import webrtcvad
from hparam import hparam as hp
from audio_process.read_wav import load_wav_torchaudio

import torch.nn.functional as F


def read_wave(path):
    """Reads a .wav file. Takes the path, and returns (PCM audio data, sample rate).
    Assumes sample width == 2
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
    return pcm_data, sample_rate  #259842 16000


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)  # 0-800,0,0.025 -> 800-1600，0.05,0.025
        timestamp += duration  # 0.025 -> 0.05
        offset += n  # 800 -> 1600

def vad_collector(sample_rate, vad, frames):
    voiced_frames = []
    for idx, frame in enumerate(frames):
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if is_speech:
            voiced_frames.append(frame)
    return b''.join(f.bytes for f in voiced_frames)

def voiced_frames_expand(voiced_frames, duration=2):
    total = duration * 16000
    expand_voiced_frames = voiced_frames
    while (len(expand_voiced_frames)) < total:
        expand_num = total - len(expand_voiced_frames)
        expand_voiced_frames += voiced_frames[: expand_num]
    return expand_voiced_frames

def vad_filter(wav_path):
    audio, sample_rate = read_wave(wav_path)
    vad = webrtcvad.Vad(3)
    frames = frame_generator(10, audio, sample_rate)
    frames = list(frames)
    voiced_frames = vad_collector(sample_rate, vad, frames)
    if len(voiced_frames) != 0:
        # Flag = True
        wave_data = np.frombuffer(voiced_frames, dtype=np.short)  # 将声音文件数据转换为数组矩阵形式
        wave_data.shape = -1, 1  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵  (n, 1)
        audio = wave_data.T  # 将矩阵转置 (1, n)
        audio = audio.reshape(-1)
    else:
        # Flag = False
        wave_data = np.frombuffer(audio, dtype=np.short)  # 将声音文件数据转换为数组矩阵形式
        wave_data.shape = -1, 1  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵  (n, 1)
        audio = wave_data.T  # 将矩阵转置 (1, n)
        audio = audio.reshape(-1)

    # 平均归一化
    # audio = (audio - audio.mean()) / (audio.max() - audio.min())
    return audio, sample_rate


if __name__ == '__main__':
    # wav_path = r'audio_process/00001.wav'
    wav_path = r'/home/ubuntu/datasets/vox1/vox1_dev_wav/wav/id10077/ILU-p2nnlyc/00001.wav'
    load_wav_torchaudio(wav_path)
    audio, sr = vad_filter(wav_path)
    print(audio.shape)
    print(sr)
    print(audio.mean(),audio.max(),audio.min())

    # t = np.arange(len(audio))
    # plt.plot(t, audio)
    # plt.show()

    # audio = (audio-audio.mean()) / (audio.max() - audio.min())
    # t = np.arange(len(audio))
    # plt.plot(t, audio)
    # plt.show()


