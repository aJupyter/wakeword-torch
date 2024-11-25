import sounddevice as sd
from scipy.io import wavfile
import time 
import numpy as np
import threading
from config import *
import librosa
from model import WakeWordModel
from audio import specgram

hello,hello_sr=librosa.load('hello.mp3')

checkpoint=torch.load('checkpoint.pth')

model=WakeWordModel().to(DEVICE)
model.load_state_dict(checkpoint['model'])
model.eval()

print(f'模型性能 -> presision:{checkpoint["best_precision"]} recall:{checkpoint["best_recall"]}')

lock=threading.Lock()
window_size=RATE*BACKGROUND_DURATION
buffer=np.array([],dtype=np.int16)

def callback(indata,frames,time,status):
    global buffer,lock,window_size

    indata=indata.copy().squeeze()
    with lock:
        buffer=np.concatenate((buffer,indata))
        if buffer.shape[0]>window_size:
            buffer=buffer[-window_size:]

with sd.InputStream(samplerate=RATE,channels=1,callback=callback,dtype='int16'):
    while True:
        segment=None
        with lock:
            if buffer.shape[0]==window_size:
                segment=buffer.copy()
        if segment is not None:
            wavfile.write('predict.wav',RATE,segment)
            
            x=specgram('predict.wav')
            x=torch.tensor(x).unsqueeze(0).type(torch.float32).to(DEVICE)
            output=model(x)
            output=output.flatten()
            pred=(output>0.9)
            
            # 只检测最后1秒是否存在激活信号
            if pred[-int(1/BACKGROUND_DURATION*OUTPUT_STEPS):].sum()>0:
                print('wakeword detected!')
                sd.play(hello,samplerate=hello_sr)
                sd.wait()
                time.sleep(WAKEWORD_DURATION) # 然后等2秒，让最后1秒语音流逝过去，下次不会再次检出

        time.sleep(0.1)