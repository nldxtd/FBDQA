from cmath import log
import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm 

import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='../log/tensorboard/',comment='_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

torch.manual_seed(1024)
np.random.seed(1024)

def feature_engineering():
    columns_need = ['bid1','bsize1',
                    'bid2','bsize2',
                    'bid3','bsize3',
                    'bid4','bsize4',
                    'bid5','bsize5',
                    'ask1','asize1',
                    'ask2','asize2',
                    'ask3','asize3',
                    'ask4','asize4',
                    'ask5','asize5',
                    'spread1','mid_price1',
                    'spread2','mid_price2',
                    'spread3','mid_price3',
                    'weighted_ab1','weighted_ab2','weighted_ab3','amount',
                    'vol1_rel_diff','volall_rel_diff','label_5','label_10','label_20','label_40','label_60', 
                ]
    dataTensor=torch.zeros(1,79,2,1999,37,dtype=torch.float)
    for sym in [4,]:
        for date in range(79):
            for timeIndex, time in enumerate(['am', 'pm']):  
                filePath = f"../data/raw/snapshot_sym{sym}_date{date}_{time}.csv"
                if not os.path.isfile(filePath):
                    print(filePath)
                    continue
                new_df = pd.read_csv(filePath)

                # region feature engineering
                # 价格+1（从涨跌幅还原到对前收盘价的比例）
                new_df['bid1'] = new_df['n_bid1']+1
                new_df['bid2'] = new_df['n_bid2']+1
                new_df['bid3'] = new_df['n_bid3']+1
                new_df['bid4'] = new_df['n_bid4']+1
                new_df['bid5'] = new_df['n_bid5']+1
                new_df['ask1'] = new_df['n_ask1']+1
                new_df['ask2'] = new_df['n_ask2']+1
                new_df['ask3'] = new_df['n_ask3']+1
                new_df['ask4'] = new_df['n_ask4']+1
                new_df['ask5'] = new_df['n_ask5']+1
        
                # 量价组合
                new_df['spread1'] =  new_df['ask1'] - new_df['bid1']
                new_df['spread2'] =  new_df['ask2'] - new_df['bid2']
                new_df['spread3'] =  new_df['ask3'] - new_df['bid3']
                new_df['mid_price1'] =  new_df['ask1'] + new_df['bid1']
                new_df['mid_price2'] =  new_df['ask2'] + new_df['bid2']
                new_df['mid_price3'] =  new_df['ask3'] + new_df['bid3']
                new_df['weighted_ab1'] = (new_df['ask1'] * new_df['n_bsize1'] + new_df['bid1'] * new_df['n_asize1']) / (new_df['n_bsize1'] + new_df['n_asize1'])
                new_df['weighted_ab2'] = (new_df['ask2'] * new_df['n_bsize2'] + new_df['bid2'] * new_df['n_asize2']) / (new_df['n_bsize2'] + new_df['n_asize2'])
                new_df['weighted_ab3'] = (new_df['ask3'] * new_df['n_bsize3'] + new_df['bid3'] * new_df['n_asize3']) / (new_df['n_bsize3'] + new_df['n_asize3'])

                new_df['relative_spread1'] = new_df['spread1'] / new_df['mid_price1']
                new_df['relative_spread2'] = new_df['spread2'] / new_df['mid_price2']
                new_df['relative_spread3'] = new_df['spread3'] / new_df['mid_price3']
                
                # 对量取对数
                new_df['bsize1'] = (new_df['n_bsize1']*10000).map(np.log1p)
                new_df['bsize2'] = (new_df['n_bsize2']*10000).map(np.log1p)
                new_df['bsize3'] = (new_df['n_bsize3']*10000).map(np.log1p)
                new_df['bsize4'] = (new_df['n_bsize4']*10000).map(np.log1p)
                new_df['bsize5'] = (new_df['n_bsize5']*10000).map(np.log1p)
                new_df['asize1'] = (new_df['n_asize1']*10000).map(np.log1p)
                new_df['asize2'] = (new_df['n_asize2']*10000).map(np.log1p)
                new_df['asize3'] = (new_df['n_asize3']*10000).map(np.log1p)
                new_df['asize4'] = (new_df['n_asize4']*10000).map(np.log1p)
                new_df['asize5'] = (new_df['n_asize5']*10000).map(np.log1p)
                new_df['amount'] = (new_df['amount_delta']/100000).map(np.log1p)
                
                new_df['vol1_rel_diff']   = (new_df['n_bsize1'] - new_df['n_asize1']) / (new_df['n_bsize1'] + new_df['n_asize1'])
                new_df['volall_rel_diff'] = (new_df['n_bsize1'] + new_df['n_bsize2'] + new_df['n_bsize3'] + new_df['n_bsize4'] + new_df['n_bsize5'] \
                                - new_df['n_asize1'] - new_df['n_asize2'] - new_df['n_asize3'] - new_df['n_asize4'] - new_df['n_asize5'] ) / \
                                ( new_df['n_bsize1'] + new_df['n_bsize2'] + new_df['n_bsize3'] + new_df['n_bsize4'] + new_df['n_bsize5'] \
                                + new_df['n_asize1'] + new_df['n_asize2'] + new_df['n_asize3'] + new_df['n_asize4'] + new_df['n_asize5'] )
                
                # endregion
                # if sym not in dataDict.keys():
                    # dataDict[sym]={}
                # if date not in dataDict[sym].keys():
                    # dataDict[sym][date]={}
                dataTensor[0][date][timeIndex]=torch.tensor(new_df[columns_need].to_numpy(),dtype=torch.float)
    # print(dataTensor.shape)
    pickle.dump(dataTensor,open('../data/pkl/1.pkl','wb'))
    return dataTensor

class Dataset(data.Dataset):
    def __init__(self, _dataTensor, labelIndex):
        self._dataTensor = _dataTensor
        self.labelIndex = labelIndex
        self.length = 0
        self.indexToSym=torch.zeros(_dataTensor.nelement(),dtype=torch.int)
        self.indexToDate=torch.zeros(_dataTensor.nelement(),dtype=torch.int)
        self.indexToTime=torch.zeros(_dataTensor.nelement(),dtype=torch.int)
        self.indexToStartIndex=torch.zeros(_dataTensor.nelement(),dtype=torch.int)
        for sym in range(self._dataTensor.shape[0]):
            for date in range(self._dataTensor.shape[1]):
                for time in range(self._dataTensor.shape[2]):
                    if torch.sum(self._dataTensor[sym][date][time]).item() != 0:
                        for startIndex in range(1900):
                            self.indexToSym[self.length]=sym
                            self.indexToDate[self.length]=date
                            self.indexToTime[self.length]=time
                            self.indexToStartIndex[self.length]=startIndex
                            self.length+=1
                    else:
                        print(sym,date,time)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        sym = self.indexToSym[index]
        date = self.indexToDate[index]
        time = self.indexToTime[index]
        startIndex = self.indexToStartIndex[index]
        data = self._dataTensor[sym][date][time][startIndex:startIndex+100,:-5]
        label = self._dataTensor[sym][date][time][startIndex+99][-5+self.labelIndex]
        # print(label)
        # print(data.shape)
        # print(label.shape)
        # fsdhjk
        return data,label.to(torch.long)

def construct_dataset(batchSize):
    if os.path.exists('../data/pkl/1.pkl'):
        dataTensor = pickle.load(open('../data/pkl/1.pkl','rb'))
    else:
        dataTensor=feature_engineering()
    dataTensor=dataTensor.to('cuda')
    # print(dataTensor.device)
    labelIndex=0 # 0 for label_5, 1 for label_10, 2 for label_20, 3 for label_40, 4 for label_60
    trainTensor=dataTensor[:,:63]
    validateTensor=dataTensor[:,63:71]
    testTensor=dataTensor[:,71:]
    # print(trainTensor.shape)
    # print(validateTensor.shape)
    # print(testTensor.shape)
    # fdshkj
    trainDataset = Dataset(_dataTensor=trainTensor,labelIndex=labelIndex)
    validateDataset=Dataset(_dataTensor=validateTensor,labelIndex=labelIndex)
    testDataset=Dataset(_dataTensor=testTensor,labelIndex=labelIndex)
    # print(testDataset)
    trainLoader = data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)
    validateLoader=data.DataLoader(dataset=validateDataset, batch_size=batchSize, shuffle=True)
    testLoader=data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=True)
    # print(testLoader.)
    return trainLoader,validateLoader,testLoader

class DeepLob(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5,1),stride=(2,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1),stride=(2,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,8)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1),stride=(2,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )        
        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        # lstm layers
        # self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        # self.fc1 = nn.Linear(64, self.y_len)
        self.fc = nn.Sequential(nn.Linear(384, 64),nn.Linear(64, self.num_classes))

    def forward(self, x):
        x=torch.unsqueeze(x,1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        x = x.reshape(-1,48*8)
        x = self.fc(x)
        forecast_y = torch.softmax(x, dim=1)
        return forecast_y

def train_one_epoch(model:nn.Module, trainLoader:data.DataLoader, criterion, optimizer, scheduler):
    model.train()
    TP=0
    TN=0
    FP=0
    FN=0
    losses=torch.zeros(len(trainLoader),dtype=torch.float)
    # for index, (input, label) in enumerate(trainDataset):
    for index,(input, label) in enumerate(tqdm(trainLoader)):
        output = model(input)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[index]=loss

        predictLabel = torch.argmax(output, axis=1)
        TP+=(torch.sum(torch.mul(predictLabel==2,label==2))+torch.sum(torch.mul(predictLabel==0,label==0)))
        TN+=torch.sum(torch.mul(predictLabel==1,label==1))
        FP+=(torch.sum(torch.mul(predictLabel==2,label!=2))+torch.sum(torch.mul(predictLabel==0,label!=0)))
        FN+=(torch.sum(torch.mul(predictLabel!=2,label==2))+torch.sum(torch.mul(predictLabel!=0,label==0)))

        # writer.add_scalars(modelName,{'trainScore':trainScore,'validateScore':validateScore,"trainLoss":trainLoss,"validateLoss":validateLoss}, epoch)
    scheduler.step()
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    beta=0.5
    f_beta=(1+beta**2)*precision*recall/(beta**2*precision+recall)
    return (torch.sum(losses)/losses.nelement()).item(),f_beta.item(),accuracy.item()

def validate_one_epoch(model:nn.Module, validateLoader:data.DataLoader, criterion):
    model.eval()
    # with torch.no_grad():
    TP=0
    TN=0
    FP=0
    FN=0
    # losses=torch.zeros(len(validateLoader),dtype=torch.float)
    # for index, (input, label) in enumerate(validateDataset):
    for (input, label) in tqdm(validateLoader):
        output = model(input)
        # loss = criterion(output, label)
        # losses[index]=loss
        predictLabel = torch.argmax(output, axis=1)
        TP+=(torch.sum(torch.mul(predictLabel==2,label==2))+torch.sum(torch.mul(predictLabel==0,label==0)))
        TN+=torch.sum(torch.mul(predictLabel==1,label==1))
        FP+=(torch.sum(torch.mul(predictLabel==2,label!=2))+torch.sum(torch.mul(predictLabel==0,label!=0)))
        FN+=(torch.sum(torch.mul(predictLabel!=2,label==2))+torch.sum(torch.mul(predictLabel!=0,label==0)))
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    beta=0.5
    f_beta=(1+beta**2)*precision*recall/(beta**2*precision+recall)
    return f_beta.item(),accuracy.item() #(torch.sum(losses)/losses.nelement()).item(),

def get_test_dataset_accuracy(model:nn.Module, testDataset):
    model.eval()    
    rightCount = 0
    totalCount = 0
    TP=0
    TN=0
    FP=0
    FN=0
    # for index, (input, label) in enumerate(testDataset):
    for index, (input, label) in enumerate(tqdm(testDataset)):
        output = model(input)
        predictLabel = torch.argmax(output, axis=1)
        rightCount += torch.sum(predictLabel == label)
        totalCount += label.shape[0]
        TP+=torch.sum(torch.mul(predictLabel,label))
        TN+=torch.sum(torch.mul(1-predictLabel,1-label))
        FP+=torch.sum(torch.mul(predictLabel,1-label))
        FN+=torch.sum(torch.mul(1-predictLabel,label))
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1=2*precision*recall/(precision+recall)
    return (rightCount / totalCount).item(),F1.item()

def trial(model:nn.Module,modelName,epochs,batchSize,savedName):
    trainLoader,validateLoader,testLoader=construct_dataset(batchSize=batchSize)

    model.to('cuda')
    # summary(model, (1, 1, 100, 32))
    
    criterion=nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 1e-5)
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=5,eta_min=1e-6)

    maxBearableEpochs=30
    noProgressEpochs=0
    stopEpoch=0
    currentBestScore=0.0
    currentBestEpoch=0
    for epoch in range(epochs):
        trainLoss,trainScore,trainAccuracy=train_one_epoch(model=model, trainLoader=trainLoader, criterion=criterion,optimizer=optimizer,scheduler=scheduler)
        validateScore,validateAccuracy=validate_one_epoch(model=model, validateLoader=validateLoader, criterion=criterion) #validateLoss,
        print("epoch:",epoch,"trainLoss:",trainLoss,"trainScore:",trainScore,"trainAccuracy:",trainAccuracy,\
            "validateLoss:",validateScore,"validateAccuracy",validateAccuracy,"delta:",validateScore-currentBestScore) # validateLoss,"validateScore:",
        writer.add_scalars(modelName,{'trainScore':trainScore,'validateScore':validateScore,"trainLoss":trainLoss,}, epoch) #"validateLoss":validateLoss
        if validateScore > currentBestScore:
            currentBestScore=validateScore
            currentBestEpoch=epoch
            torch.save(model, "../weight/"+modelName+"/"+savedName)
            noProgressEpochs=0
        else:
            noProgressEpochs+=1
            if noProgressEpochs>=maxBearableEpochs:
                stopEpoch=epoch
                break
        stopEpoch=epoch
    testScore=get_test_dataset_accuracy(model,testDataset=testLoader)
    print("==========================================================================================")
    print("testScore",testScore,"validateScore",currentBestScore,"bestEpoch",currentBestEpoch,"stopEpoch",stopEpoch)
    print("==========================================================================================")

if __name__=='__main__':
    model=DeepLob(num_classes=3)
    trial(model=model,modelName="DeepLob",epochs=200,batchSize=512,savedName="1.pth")

    print("All is well!")
    writer.close()


