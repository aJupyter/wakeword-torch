from audio import record,collect
import os
from config import *
import time

nowakeword_list=collect(NOWAKEWORD_DIR)
print(f'当前nowakeword音频数量:{len(nowakeword_list)}')

rec_times=int(input('输入希望录制的音频数量:'))
for i in range(rec_times):
    input(f'{i+1}条,回车开始...')
    record(f'{NOWAKEWORD_DIR}/{int(time.time())}.wav',NOWAKEWORD_DURATION)