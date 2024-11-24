import torch 
from dataset import WakeWordDataset
from model import WakeWordModel
from config import * 
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

if __name__=='__main__':
    checkpoint=None
    try:
        checkpoint=torch.load('checkpoint.pth')
    except:
        pass
    
    model=WakeWordModel().to(DEVICE)
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    train_dataset=WakeWordDataset()
    train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)

    test_dataset=WakeWordDataset()
    test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)
    
    loss_fn=torch.nn.BCELoss()

    tensorboard=SummaryWriter(log_dir=f'runs/wakeword_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    best_precision=None
    best_recall=None
    if checkpoint:
        best_precision=checkpoint['best_precision']
        best_recall=checkpoint['best_recall']
    
    for epoch in range(EPOCH):
        model.train()
        for x,y in train_dataloader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            output=model(x)
            loss=loss_fn(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        TP=0
        FP=0
        FN=0
        model.eval()
        for x,y in test_dataloader:
            with torch.no_grad():
                x,y=x.to(DEVICE),y.to(DEVICE)
                output=model(x)
                output=output.flatten()
                y=y.flatten()
                
                preds=(output>0.5).type(torch.int)
                y=(y>0.5).type(torch.int)
                
                # 计算Accuracy和Precision
                TP+=((preds==1) & (y==1)).sum().item()
                FP+=((preds==1) & (y==0)).sum().item()
                FN+=((preds==0) & (y==1)).sum().item()
        
        precision=TP/(TP+FP) if TP+FP!=0 else 0
        recall=TP/(TP+FN) if TP+FN!=0 else 0
        tensorboard.add_scalar('metrics/train_loss',loss.item(),epoch)
        tensorboard.add_scalar('metrics/test_precision',precision,epoch)
        tensorboard.add_scalar('metrics/test_recall',recall,epoch)
        print(f'epoch:{epoch} train_loss:{loss.item()} TP:{TP} FP:{FP} FN:{FN} test_precision:{precision} test_recall:{recall}')
        
        if best_precision is None or (best_precision<precision and best_recall<recall):
            best_precision,best_recall=precision,recall
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'best_precision':best_precision,'best_recall':best_recall},'.checkpoint.pth')
            os.replace('.checkpoint.pth','checkpoint.pth')