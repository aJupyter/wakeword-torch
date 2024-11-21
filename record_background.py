from audio import record,collect
import os
from config import *
import time

background_list=collect(BACKGROUND_DIR)
print(f'当前背景音频数量:{len(background_list)}')

rec_times=int(input('输入希望录制的音频数量:'))
for i in range(rec_times):
    print(f'{i+1}条录制中...')
    record(f'{BACKGROUND_DIR}/{int(time.time())}.wav',BACKGROUND_DURATION)