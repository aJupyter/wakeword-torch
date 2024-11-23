import torch 
from config import * 
import json
import numpy as np

class WakeWordDataset(torch.utils.data.Dataset):
    def __init__(self,is_train=True):
        super().__init__()
        
        with open(TRAIN_LABELS if is_train else TEST_LABELS,'r') as fp:
            self.data=json.load(fp)

    def __getitem__(self,index):
        sample=np.load(self.data[index]['sample'])
        return torch.tensor(sample['x'],dtype=torch.float32),torch.tensor(sample['y'],dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
if __name__=="__main__":
    ds=WakeWordDataset()
    x,y=ds[0]
    print(x.shape,y.shape)