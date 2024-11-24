from config import *
from audio import load,merge,collect,specgram
import random
import numpy as np
import os 
import json

def random_insert_audio(audio_list,is_wakeword,seg_list,background,y,max_retry=20):
    audio=load(random.choice(audio_list))
    for _ in range(max_retry):
        audio_beg=random.randint(0,len(background)-len(audio))
        audio_end=audio_beg+len(audio)
        ok=True
        for seg in seg_list:
            if not (audio_end<seg[0] or seg[1]<audio_beg):
                ok=False
                break
        if ok:
            seg_list.append((audio_beg,audio_end,audio))
            background=merge(background,audio,audio_beg)
            if is_wakeword:
                y_pos=int(audio_end/len(background)*OUTPUT_STEPS)
                y[y_pos:y_pos+50,0]=1
            break
    return background
    
def generate_dataset(datadir,manifest,samples):
    os.makedirs(datadir,exist_ok=True)
    
    background_list=collect(BACKGROUND_DIR)
    wakeword_list=collect(WAKEWORD_DIR)
    nowakeword_list=collect(NOWAKEWORD_DIR)
    
    dataset=[]
    
    for i in range(samples):
        background=load(random.choice(background_list))
        y=np.zeros((OUTPUT_STEPS,1))
        
        seg_list=[]
        for _ in range(random.randint(0,4)):
            if random.random()<0.5:
                background=random_insert_audio(audio_list=wakeword_list,is_wakeword=True,seg_list=seg_list,background=background,y=y)
            else:
                background=random_insert_audio(audio_list=nowakeword_list,is_wakeword=False,seg_list=seg_list,background=background,y=y)     
        
        wav_filename=f'{datadir}/{i}.wav'
        background.export(wav_filename,format='wav')

        x=specgram(wav_filename)
        sample_filename=f'{datadir}/{i}.npz'
        np.savez(sample_filename,x=x,y=y)
        
        dataset.append({'wav':wav_filename,'sample':sample_filename})

        print(f'{wav_filename} 生成完毕')
    with open(manifest,'w') as fp:
        json.dump(dataset,fp)
        
if __name__=='__main__':
    train_samples=int(input('输入希望生成的训练集大小:'))
    test_samples=int(input('输入希望生成的测试集大小:'))
    generate_dataset(TRAIN_DIR,TRAIN_LABELS,train_samples)
    generate_dataset(TEST_DIR,TEST_LABELS,test_samples)