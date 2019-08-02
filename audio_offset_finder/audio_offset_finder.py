# audio-offset-finder
#
# Copyright (c) 2014 British Broadcasting Corporation
# Copyright (c) 2018 Abram Hindle
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from subprocess import Popen, PIPE
from scipy.io import wavfile
# from scikits.talkbox.features.mfcc import mfcc
import matplotlib.pyplot as plt
import librosa
import os, tempfile, warnings
import numpy as np

def mfcc(audio, nwin=256, nfft=512, fs=16000, nceps=13):
    #return librosa.feature.mfcc(y=audio, sr=44100, hop_length=nwin, n_mfcc=nceps)
    return [np.transpose(librosa.feature.mfcc(y=audio, sr=fs, n_fft=nfft, win_length=nwin,n_mfcc=nceps))]

def add_feature(mfcc1, rmsa1):
    tmfcc1 = np.zeros((mfcc1.shape[0],mfcc1.shape[1]+rmsa1.shape[0]))
    n = mfcc1.shape[0]
    m = mfcc1.shape[1]
    w = rmsa1.shape[0]
    tmfcc1[0:n,0:m] = mfcc1[0:n,0:m]
    tmfcc1[0:n,m:m+w]   = np.transpose(rmsa1[0:w,0:n])
    return tmfcc1

def find_offset(file1, file2, fs=8000, trim=60*15, correl_nframes=1000, plotit=True):
    sr = fs
    tmp1 = convert_and_trim(file1, fs, trim)
    tmp2 = convert_and_trim(file2, fs, trim)
    # Removing warnings because of 18 bits block size
    # outputted by ffmpeg
    # https://trac.ffmpeg.org/ticket/1843
    warnings.simplefilter("ignore", wavfile.WavFileWarning)
    a1 = wavfile.read(tmp1, mmap=True)[1] / (2.0 ** 15)
    a2 = wavfile.read(tmp2, mmap=True)[1] / (2.0 ** 15)
    # We truncate zeroes off the beginning of each signals
    # (only seems to happen in ffmpeg, not in sox)
    a1 = ensure_non_zero(a1)
    a2 = ensure_non_zero(a2)
    print("Ref samples: %s Find samples: %s" % (a1.shape[0],a2.shape[0]))
    mfcc1 = mfcc(a1, nwin=256, nfft=512, fs=fs, nceps=26)[0]
    mfcc2 = mfcc(a2, nwin=256, nfft=512, fs=fs, nceps=26)[0]
    mfcc1 = std_mfcc(mfcc1)
    mfcc2 = std_mfcc(mfcc2)
    rmsa1 = librosa.feature.rms(a1)
    rmsa2 = librosa.feature.rms(a2)
    cent1 = librosa.feature.spectral_centroid(y=a1, sr=fs)
    cent2 = librosa.feature.spectral_centroid(y=a2, sr=fs)
    rolloff1 = librosa.feature.spectral_rolloff(y=a1, sr=fs, roll_percent=0.1)
    rolloff2 = librosa.feature.spectral_rolloff(y=a2, sr=fs, roll_percent=0.1)
    chroma_cq1 = librosa.feature.chroma_cqt(y=a1, sr=fs, n_chroma=10)
    chroma_cq2 = librosa.feature.chroma_cqt(y=a2, sr=fs, n_chroma=10)
    
    onset_env1 = librosa.onset.onset_strength(y=a1, sr=sr)
    onset_env2 = librosa.onset.onset_strength(y=a2, sr=sr)
    pulse1 = librosa.beat.plp(onset_envelope=onset_env1, sr=sr)
    pulse2 = librosa.beat.plp(onset_envelope=onset_env2, sr=sr)


    mfcc1 = add_feature(mfcc1, rmsa1)
    mfcc1 = add_feature(mfcc1, rolloff1/fs)
    mfcc1 = add_feature(mfcc1, cent1/fs)
    mfcc1 = add_feature(mfcc1, chroma_cq1)
    mfcc1 = add_feature(mfcc1, onset_env1.reshape(1,onset_env1.shape[0]))
    mfcc1 = add_feature(mfcc1, pulse1.reshape(1,onset_env1.shape[0]))

    mfcc2 = add_feature(mfcc2, rmsa2)
    mfcc2 = add_feature(mfcc2, rolloff2/fs)
    mfcc2 = add_feature(mfcc2, cent2/fs)
    mfcc2 = add_feature(mfcc2, chroma_cq2)
    mfcc2 = add_feature(mfcc2, onset_env2.reshape(1,onset_env2.shape[0]))
    mfcc2 = add_feature(mfcc2, pulse2.reshape(1,onset_env2.shape[0]))

    c = cross_correlation(mfcc1, mfcc2, nframes=correl_nframes)

    max_k_index = np.argmax(c)
    # # The MFCC window overlap is hardcoded in scikits.talkbox
    # # offset = max_k_index * 160.0 / float(fs) # * over / sample rate
    offset = max_k_index * (a1.shape[0]/rmsa1.shape[1]) / float(fs) # * over / sample rate
    print("Best matching window: %s" % max_k_index)
    score = (c[max_k_index] - np.mean(c)) / np.std(c) # standard score of peak
    os.remove(tmp1)
    os.remove(tmp2)
    if plotit:
        plt.figure(figsize=(8, 4))
        plt.plot(c)
        plt.show()
    return offset, score

def ensure_non_zero(signal):
    # We add a little bit of static to avoid
    # 'divide by zero encountered in log'
    # during MFCC computation
    signal += np.random.random(len(signal)) * 10**-10
    return signal

def make_similar_shape(mfcc1,mfcc2):
    n1, mdim1 = mfcc1.shape
    n2, mdim2 = mfcc2.shape
    # print((nframes,(n1,mdim1),(n2,mdim2)))
    if (n2 < n1):
        t = np.zeros((n1,mdim2))
        t[0:n2,0:mdim2] = mfcc2[0:n2,0:mdim2]
        mfcc2 = t
    elif (n2 > n1):
        return make_similar_shape(mfcc2,mfcc1)
    return (mfcc1,mfcc2)

def cross_correlation(mfcc1, mfcc2, nframes):
    n1, mdim1 = mfcc1.shape
    n2, mdim2 = mfcc2.shape
    # print((nframes,(n1,mdim1),(n2,mdim2)))
    if (n2 < nframes):
        t = np.zeros((nframes,mdim2))
        t[0:n2,0:mdim2] = mfcc2[0:n2,0:mdim2]
        mfcc2 = t
    n = n1 - nframes + 1
    #c = np.zeros(min(n2,n))
    c = np.zeros(n)
    #for k in range(min(n2,n)):
    for k in range(n):
        cc = np.sum(np.multiply(mfcc1[k:k+nframes], mfcc2[:nframes]), axis=0)
        c[k] = np.linalg.norm(cc,1)
    return c

def std_mfcc(mfcc):
    return (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)

def convert_and_trim(afile, fs, trim):
    tmp = tempfile.NamedTemporaryFile(mode='r+b', prefix='offset_', suffix='.wav')
    tmp_name = tmp.name
    tmp.close()
    psox = Popen([
        'ffmpeg', '-loglevel', 'panic', '-i', afile, 
        '-ac', '1', '-ar', str(fs), '-ss', '0', '-t', str(trim), 
        '-acodec', 'pcm_s16le', tmp_name
    ], stderr=PIPE)
    psox.communicate()
    if not psox.returncode == 0:
        raise Exception("FFMpeg failed")
    return tmp_name
