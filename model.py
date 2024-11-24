import torch 
from config import *

class WakeWordModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_seq=torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=FREQ_SIZE,out_channels=HIDDEN_SIZE,kernel_size=15,stride=4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
        )
        self.lstm=torch.nn.LSTM(input_size=HIDDEN_SIZE,hidden_size=HIDDEN_SIZE,num_layers=2,batch_first=True)
        self.head=torch.nn.Sequential(
            torch.nn.Linear(in_features=HIDDEN_SIZE,out_features=1),
            torch.nn.Dropout(0.1),
            torch.nn.Sigmoid(),
        )
        
    def forward(self,x):
        y=self.conv_seq(x)
        y=y.permute(0,2,1)
        y,(_,_)=self.lstm(y)
        y=self.head(y)
        return y
    
if __name__=='__main__':
    model=WakeWordModel()
    x=torch.randn((5,101,5511))
    y=model(x)
    print(y)