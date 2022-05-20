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

torch.manual_seed(1024)
np.random.seed(1024)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# ### 一、数据处理与特征工程
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
    print(dataTensor.shape)
    return dataTensor.to('cuda')
    # print(dataTensor.element_size())
    # print(dataTensor.nelement())
    # print(dataTensor.element_size()*dataTensor.nelement())

def construct_dataset(dataTensor):
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
            return data, label

    batch_size = 512
    labelIndex=0 # 0 for label_5, 1 for label_10, 2 for label_20, 3 for label_40, 4 for label_60
    trainTensor=dataTensor[:,:63]
    validateTensor=dataTensor[:,63:71]
    testTensor=dataTensor[:,71:]
    trainDataset = Dataset(_dataTensor=trainTensor,labelIndex=labelIndex)
    validateDataset=Dataset(_dataTensor=validateTensor,labelIndex=labelIndex)
    testDataset=Dataset(_dataTensor=testTensor,labelIndex=labelIndex)
    # print(testDataset)
    trainLoader = data.DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True)
    validateLoader=data.DataLoader(dataset=validateDataset, batch_size=batch_size, shuffle=True)
    testLoader=data.DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=True)
    # print(testLoader.)
    return trainLoader,validateLoader,testLoader

class Deeplob(nn.Module):
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

def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    for it in tqdm(range(epochs)):
        if ((epochs+1) % 10 == 0):
            optimizer.lr = optimizer.lr*0.5
        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()
            
            optimizer.step()
            
            train_loss.append(loss.item())
            
        # Get train loss and test loss
        train_loss = np.mean(train_loss) # a little misleading
    
        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)      
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss
        
        if test_loss < best_test_loss:
            torch.save(model, f'../weight/deeplob/best_val_model_pytorch_sym{sym}_date{dates[-1]}')
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')

        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')
    torch.save(model, f'../weight/deeplob/final_model_pytorch_sym{sym}_date{dates[-1]}')
    return train_losses, test_losses

def trial():
    model = Deeplob(num_classes = 3)
    model.to(device)
    summary(model, (1, 1, 100, 32))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 1e-5)

    train_losses, val_losses = batch_gd(model, criterion, optimizer, train_loader1, val_loader1, epochs=50)

if __name__=='__main__':
    dataTensor=feature_engineering()
    construct_dataset(dataTensor=dataTensor)
    # trial()
    print("All is well!")


