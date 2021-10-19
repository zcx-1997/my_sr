# -*- coding: utf-8 -*-

"""
@Time : 2021/10/19
@Author : Lenovo
@File : vad_segment
@Description : 
"""
import time
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
    data, sr = librosa.load(path, sample_rate)
    assert len(data.shape) == 1
    assert sr in (8000, 16000, 32000, 48000)

    return data, pcm_data, sample_rate  #259842 16000


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

def vad_collector_pad(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as reported by the VAD),
    the collector triggers and begins yielding audio frames.
    Then the collector waits until 90% of the frames in the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                start = ring_buffer[0][0].timestamp
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield (start, frame.timestamp + frame.duration)
                ring_buffer.clear()
                voiced_frames = []
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield (start, frame.timestamp + frame.duration)


def vad_chunk(audio, voiced_times,sample_rate=16000):
    speech_times = []
    speech_segs = []
    for i, voiced_time in enumerate(voiced_times):
        start = np.round(voiced_time[0], decimals=2)
        end = np.round(voiced_time[1], decimals=2)
        j = start
        while j + .4 < end:
            end_j = np.round(j + .4, decimals=2)
            speech_times.append((j, end_j))
            speech_segs.extend(audio[int(j * sample_rate):int(end_j * sample_rate)])
            j = end_j
        else:
            speech_times.append((j, end))
            speech_segs.extend(audio[int(j * sample_rate):int(end * sample_rate)])

        # voiced_segs = [x for seg in speech_segs for x in seg]
        audio = np.array(speech_segs)
        return audio, sample_rate

def vad_filter(wav_path):
    _, signal, sample_rate = read_wave(wav_path)
    vad = webrtcvad.Vad(hp.vad.mode)
    frames = frame_generator(hp.vad.frame_duration_ms, signal, sample_rate)
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
        wave_data = np.frombuffer(signal, dtype=np.short)  # 将声音文件数据转换为数组矩阵形式
        wave_data.shape = -1, 1  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵  (n, 1)
        audio = wave_data.T  # 将矩阵转置 (1, n)
        audio = audio.reshape(-1)

    # 平均归一化
    # audio = (audio - audio.mean()) / (audio.max() - audio.min())
    return audio, sample_rate
def vad_filter_pad(wav_path):
    signal, byte_signal, sample_rate = read_wave(wav_path)
    vad = webrtcvad.Vad(hp.vad.mode)
    frames = frame_generator(hp.vad.frame_duration_ms, byte_signal, sample_rate)
    frames = list(frames)

    voiced_times = vad_collector_pad(sample_rate, hp.vad.frame_duration_ms, hp.vad.padding_duration_ms, vad, frames)
    try:
        next(voiced_times)
    except:
        flag = 0
    else:
        flag = 1
    if flag != 0:
        voiced_times = vad_collector_pad(sample_rate, hp.vad.frame_duration_ms, hp.vad.padding_duration_ms, vad, frames)
        audio, sr = vad_chunk(signal,voiced_times,sample_rate)
    else:
        audio, sr = signal, sample_rate
    return audio, sr


if __name__ == '__main__':
    wav_path = r'audio_process/00001.wav'
    # wav_path = r'/home/ubuntu/datasets/vox1/vox1_dev_wav/wav/id10077/ILU-p2nnlyc/00001.wav'
    load_wav_torchaudio(wav_path)
    print("=========no pad:")
    start = time.time()
    audio, sr = vad_filter(wav_path)
    print("time cost",time.time()-start)
    print(audio.shape)
    print(sr)
    print(audio.mean(),audio.max(),audio.min())

    t = np.arange(len(audio))
    plt.plot(t, audio)
    plt.show()

    print("========pad:")
    start = time.time()
    audio, sr = vad_filter_pad(wav_path)
    print("time cost", time.time() - start)
    print(audio.shape)
    print(sr)
    print(audio.mean(),audio.max(),audio.min())

    t = np.arange(len(audio))
    plt.plot(t, audio)
    plt.show()

