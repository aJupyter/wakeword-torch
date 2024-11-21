from audio import record,collect
import os
from config import *
import time

wakeword_list=collect(WAKEWORD_DIR)
print(f'当前wakeword音频数量:{len(wakeword_list)}')

rec_times=int(input('输入希望录制的音频数量:'))
for i in range(rec_times):
    input(f'{i+1}条,回车开始...')
    record(f'{WAKEWORD_DIR}/{int(time.time())}.wav',WAKEWORD_DURATION)