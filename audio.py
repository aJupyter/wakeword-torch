import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt
from config import *
import os 
from pydub import AudioSegment

# 录音
def record(filename,seconds):
    audio=sd.rec(int(seconds*RATE),samplerate=RATE,channels=1)
    sd.wait()
    wavfile.write(filename,RATE,audio)

# 扫描文件
def collect(dirname):
    os.makedirs(dirname,exist_ok=True)
    
    filelist=[]
    for root,_,files in os.walk(dirname):
        for file in files:
            filename=os.path.join(root,file)
            filelist.append(filename)
    return filelist

# 加载音频
def load(filename):
    return AudioSegment.from_wav(filename)

# 合成音频
def merge(background,wakeword_list,nowakeword_list):
    for wakeword,position in wakeword_list:
        background=background.overlay(wakeword,position=position)
    for nowakeword,position in nowakeword_list:
        background=background.overlay(nowakeword,position=position)
    return background

# 提取频谱
def specgram(filename):
    rate,data=wavfile.read(filename)
    pxx,_,_,_=plt.specgram(data,NFFT=200,Fs=8000,noverlap=120) # 10s音频，返回pxx为(101,5511), 即5511个时间步, 每一步由101个频段表达
    return pxx

if __name__=='__main__':
    # pxx=specgram('data/nowakeword/1732191745.wav')
    # print(pxx.shape)
    # plt.show()
    #record('test.wav',10)
    pass