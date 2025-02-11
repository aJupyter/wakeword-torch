import torch 

RATE=44100

WAKEWORD_DURATION=2
NOWAKEWORD_DURATION=2
BACKGROUND_DURATION=10

OUTPUT_STEPS=1375 # 模型输出steps
INPUT_STEPS=5511 # 模型输入steps
FREQ_SIZE=101 # 每个step的频率个数
HIDDEN_SIZE=128 # 隐藏层

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE=32
EPOCH=400

WAKEWORD_DIR='data/wakeword'
NOWAKEWORD_DIR='data/nowakeword'
BACKGROUND_DIR='data/background'

TRAIN_DIR='data/train'
TRAIN_LABELS='data/train.json'
TEST_DIR='data/test'
TEST_LABELS='data/test.json'