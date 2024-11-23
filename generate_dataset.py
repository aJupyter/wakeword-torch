from config import *
from audio import load,merge,collect,specgram
import random
import numpy as np
import os 
import json

def generate_dataset(datadir,manifest,samples):
    os.makedirs(datadir,exist_ok=True)
    
    background_list=collect(BACKGROUND_DIR)
    wakeword_list=collect(WAKEWORD_DIR)
    nowakeword_list=collect(NOWAKEWORD_DIR)
    
    dataset=[]
    
    for i in range(samples):
        # 随机选1组音频
        background=load(random.choice(background_list))
        wakeword=load(random.choice(wakeword_list))
        nowakeword=load(random.choice(nowakeword_list))
        
        # 随机生成位置
        wakeword_beg=random.randint(0,len(background)-len(wakeword))
        wakeword_end=wakeword_beg+len(wakeword)
        while True:
            nowakeword_beg=random.randint(0,len(background)-len(nowakeword))
            nowakeword_end=nowakeword_beg+len(nowakeword)
            if wakeword_end<nowakeword_beg or nowakeword_end<wakeword_beg:
                break
                
        # 合成音频
        background=merge(background,[(wakeword,wakeword_beg)],[(nowakeword,nowakeword_beg)])
        wav_filename=f'{datadir}/{i}.wav'
        background.export(wav_filename,format='wav')

        # 特征生成
        x=specgram(wav_filename)
        y=np.zeros((OUTPUT_STEPS,1))
        y_pos=int(wakeword_end/len(background)*OUTPUT_STEPS)
        y[y_pos:y_pos+50,0]=1
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