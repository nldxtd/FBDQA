import random
import numpy as np
import pdb
import pickle
from matplotlib import pyplot as plot
from tqdm import tqdm
import json
import pandas as pd

import torch
from torch.utils.tensorboard.writer import SummaryWriter

writer = SummaryWriter('../log/tensorboard/')
np.random.seed(1024)
random.seed(1024)
torch.manual_seed(1024)

class FC(torch.nn.Module):
    def __init__(self,inputDim):
        super(FC, self).__init__()
        self.net=torch.nn.Sequential(
            # torch.nn.BatchNorm1d(714),
            torch.nn.Linear(inputDim, 1),
            torch.nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self,inputDim):
        self.network=FC(inputDim=inputDim).to('cuda')
        self.fitness=torch.tensor([0])

def generate_agents(populationSize,inputDim):
    return [Agent(inputDim=inputDim) for _ in range(populationSize)]

def calculate_fitness(agents,x,y,threshold):
    for agent in agents:
        agent:Agent
        predictY=agent.network(x)
        sortedPredictY, indices=torch.sort(predictY,dim=0,descending=True)
        fitness=0
        for item in indices[:int(threshold*len(indices))]:
            if y[item[0]][0]==1:
                fitness+=1
        agent.fitness=torch.tensor([fitness])
    return agents

def selection(agents):
    agents=sorted(agents, key=lambda agent: agent.fitness.item(),reverse=True)
    selectedAgentList=[]
    for index,agent in enumerate(agents):
        if random.random()<=(1-index/len(agents)) and random.random()<=(1-index/len(agents)):
            selectedAgentList.append(agent)
    return selectedAgentList

def mutation(agents:list,populationSize,inputDim):
    geneMutationRate=0.1
    offSpring = []
    for _ in range(populationSize-len(agents)):
        parent:Agent=agents[random.randint(0,len(agents)-1)]
        child=Agent(inputDim=inputDim)
        parentStateDict=parent.network.state_dict()
        childStateDict=child.network.state_dict()
        for name, param in parentStateDict.items():
            transformedParam=param*torch.randn(param.shape).to('cuda')*(torch.rand(param.shape)<geneMutationRate).to('cuda')+param
            childStateDict[name].copy_(transformedParam)
        offSpring.append(child)
    agents.extend(offSpring)
    return agents 

def evaluate_agent(agent:Agent,x,y,threshold):
    predictY=agent.network(x)
    sortedPredictY=sorted(enumerate(predictY), key=lambda item: item[1][0],reverse=True)
    fitness=0
    for item in sortedPredictY[:int(threshold*len(sortedPredictY))]:
        if y[item[0]][0]==1:
            fitness+=1
    return fitness

def load_data(datasetName,datasetVersion):
    data=pickle.load(open('../data/'+datasetName+'/'+datasetVersion+'/data.pkl', 'rb'))
    trainX=torch.tensor(data['trainX'],dtype=torch.float32).to('cuda')
    trainY=torch.tensor(data['trainY'],dtype=torch.float32).to('cuda')
    testX=torch.tensor(data['testX'],dtype=torch.float32).to('cuda')
    testY=torch.tensor(data['testY'],dtype=torch.float32).to('cuda')
    return trainX,trainY,testX,testY

def run_genetic_algorithm(datasetName,datasetVersion,modelVersion):
    trainX,trainY,testX,testY=load_data(datasetName=datasetName,datasetVersion=datasetVersion)
    if datasetName=='liver':
        ratio=3
    elif datasetName=='skin':
        ratio=2
    threshold=ratio*(sum(trainY==1)/trainY.shape[0]).item()
    print("threshold",threshold)
    inputDim=trainX.shape[1]
    populationSize=100
    generations=1000
    agents=generate_agents(populationSize=populationSize,inputDim=inputDim)
    currentBsetFitness=0
    for i in range(generations):
        agents=calculate_fitness(agents,trainX,trainY,threshold=threshold)
        agents=selection(agents)
        agents=mutation(agents,populationSize,inputDim=inputDim)
        if i%2==0:
            trianFitness=agents[0].fitness.item()
            testFitness=evaluate_agent(agent=agents[0],x=testX,y=testY,threshold=threshold)
            writer.add_scalars("fitness",{"train":trianFitness,"test":testFitness}, i)
            print('Generation',i,"trainFitness",trianFitness,"/",sum(trainY==1).item(),"testFitness",testFitness,'/',sum(testY==1).item())
            if trianFitness>currentBsetFitness:
                currentBsetFitness=trianFitness
                torch.save(agents[0].network.state_dict(),'../weight/'+datasetName+'/'+datasetVersion+'/'+modelVersion+'.pth')            
    
def create_dataset(datasetName,datasetVersion):
    data:dict=pickle.load(open('../data/'+datasetName+'/'+datasetVersion+'/raw.pkl', 'rb'))
    compoundNameList=list(data['all'].keys())
    random.shuffle(compoundNameList)
    all={}
    if datasetName=='liver':
        for compound in compoundNameList:
            label=0
            if compound in data['c50'].keys():
                label=1
            elif compound in data['dmso'].keys():
                label=2
            all[compound]={'label':label,'logcpm-':data['all'][compound]['cpm-'],}
    elif datasetName=='skin':
        for compound in compoundNameList:
            label=0
            if compound in data['PLATE']['pos']:
                if 'DMSO' in compound:
                    label=5
                else:
                    label=4
            else:
                if compound in data['pc'].keys():
                    label=1
                elif compound in data['nc'].keys():
                    label=2
                elif compound in data['nf'].keys():
                    label=3
            all[compound]={'label':label,'logcpm-':data['all'][compound]['logcpm-'],}
    trainX=[];trainY=[]
    testX=[];testY=[]
    for compound in list(all.keys())[:len(all.keys())//2]:
        trainX.append(all[compound]['logcpm-'])
        trainY.append([all[compound]['label']])
    for compound in list(all.keys())[len(all.keys())//2:]:
        testX.append(list(all[compound]['logcpm-']))
        testY.append([all[compound]['label']])
    trainX=np.array(trainX)
    trainY=np.array(trainY)
    testX=np.array(testX)
    testY=np.array(testY)
    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)
    if datasetName=='liver':
        print('train',sum(trainY==[0]),sum(trainY==[1]),sum(trainY==[2]))
        print('test',sum(testY==[0]),sum(testY==[1]),sum(testY==[2]))
    elif datasetName=='skin':
        print('train',sum(trainY==[0]),sum(trainY==[1]),sum(trainY==[2]),sum(trainY==[3]),sum(trainY==[4]),sum(trainY==[5]))
        print('test',sum(testY==[0]),sum(testY==[1]),sum(testY==[2]),sum(testY==[3]),sum(testY==[4]),sum(testY==[5]))
    pickle.dump({'trainX':trainX,'trainY':trainY,'testX':testX,'testY':testY},open('../data/'+datasetName+'/'+datasetVersion+'/data.pkl','wb'))

def get_compound_score(datasetName,datasetVersion,modelVersion,x):
    if datasetName=='liver':
        inputDim=52
    elif datasetName=='skin':
        inputDim=1130
    model=FC(inputDim=inputDim).to('cuda')
    model.load_state_dict(torch.load('../weight/'+datasetName+'/'+datasetVersion+'/'+modelVersion+'.pth'))
    model.eval()
    score=model(x)
    return score

def draw_scatter(datasetName,datasetVersion,modelVersion):
    trainX,trainY,testX,testY=load_data(datasetName=datasetName,datasetVersion=datasetVersion)
    # toDrawX=trainX;toDrawY=trainY
    # toDrawX=testX;toDrawY=testY
    toDrawX=torch.cat((trainX,testX),dim=0);toDrawY=torch.cat((trainY,testY),dim=0)
    score=torch.squeeze(get_compound_score(datasetName=datasetName,datasetVersion=datasetVersion,modelVersion=modelVersion,x=toDrawX)).tolist()
    label=torch.squeeze(toDrawY).tolist()
    colour=[]
    for i in label:
        if i==0:
            colour.append('white')
            # colour.append('blue')
        elif i==1:
            # colour.append('white')
            colour.append('red')
        elif i==2:
            # colour.append('white')
            colour.append('green')
        elif i==3:
            colour.append('white')
            # colour.append('orange')
        elif i==4:
            colour.append('white')
            # colour.append('brown')
        elif i==5:
            colour.append('white')
            # colour.append('black')
    plot.scatter([i for i in range(len(score))],score,s=1,label="compound",c=colour)
    # plot.legend(loc='lower right')
    plot.savefig('../result/'+datasetName+'/'+datasetVersion+'/'+modelVersion+'_scatter.png')

def save_score_list_to_excel(datasetName,datasetVersion,modelVersion,scoreList,fileName):
    scoreList.sort(key=lambda x:x[1],reverse=True)
    scoreDf=pd.DataFrame(scoreList,columns=['name','score'])
    scoreDf.to_excel('../result/'+datasetName+'/'+datasetVersion+'/'+modelVersion+'_'+fileName+'.xlsx')

def score_all_compound(datasetName,datasetVersion,modelVersion):
    data:dict=pickle.load(open('../data/'+datasetName+'/'+datasetVersion+'/raw.pkl', 'rb'))
    compoundNameList=list(data['all'].keys())
    if datasetName=='liver':
        compoundScoreList=[]
        c50ScoreList=[]
        dmsoScoreList=[]
        for compound in tqdm(compoundNameList):
            x=torch.unsqueeze(torch.tensor(data['all'][compound]['cpm-'],dtype=torch.float32),0).to('cuda')
            score=torch.squeeze(get_compound_score(datasetName=datasetName,datasetVersion=datasetVersion,modelVersion=modelVersion,x=x)).item()
            if compound in data['c50'].keys():
                c50ScoreList.append((compound,score))
            elif compound in data['dmso'].keys():
                dmsoScoreList.append((compound,score))
            else:
                compoundScoreList.append((compound,score))
        save_score_list_to_excel(datasetName=datasetName,datasetVersion=datasetVersion,modelVersion=modelVersion,scoreList=compoundScoreList,fileName='compoundScore')
        save_score_list_to_excel(datasetName=datasetName,datasetVersion=datasetVersion,modelVersion=modelVersion,scoreList=c50ScoreList,fileName='c50Score')
        save_score_list_to_excel(datasetName=datasetName,datasetVersion=datasetVersion,modelVersion=modelVersion,scoreList=dmsoScoreList,fileName='dmsoScore')
    elif datasetName=='skin':
        compoundScoreList=[]
        pcScoreList=[]
        ncScoreList=[]
        nfScoreList=[]
        posInPosPlateScoreList=[]
        dmsoInPosPlateScoreList=[]
        for compound in tqdm(compoundNameList):
            x=torch.unsqueeze(torch.tensor(data['all'][compound]['logcpm-'],dtype=torch.float32),0).to('cuda')
            score=torch.squeeze(get_compound_score(datasetName=datasetName,datasetVersion=datasetVersion,modelVersion=modelVersion,x=x)).item()
            if compound in data['PLATE']['pos']:
                if 'DMSO' in compound:
                    dmsoInPosPlateScoreList.append((compound,score))
                else:
                    posInPosPlateScoreList.append((compound,score))
            else:
                if compound in data['pc'].keys():
                    pcScoreList.append((compound,score))
                elif compound in data['nc'].keys():
                    ncScoreList.append((compound,score))
                elif compound in data['nf'].keys():
                    nfScoreList.append((compound,score))
                else:
                    compoundScoreList.append((compound,score))
        save_score_list_to_excel(datasetName=datasetName,datasetVersion=datasetVersion,modelVersion=modelVersion,scoreList=compoundScoreList,fileName='compoundScore')
        save_score_list_to_excel(datasetName=datasetName,datasetVersion=datasetVersion,modelVersion=modelVersion,scoreList=pcScoreList,fileName='pcScore')
        save_score_list_to_excel(datasetName=datasetName,datasetVersion=datasetVersion,modelVersion=modelVersion,scoreList=ncScoreList,fileName='ncScore')
        save_score_list_to_excel(datasetName=datasetName,datasetVersion=datasetVersion,modelVersion=modelVersion,scoreList=nfScoreList,fileName='nfScore')
        save_score_list_to_excel(datasetName=datasetName,datasetVersion=datasetVersion,modelVersion=modelVersion,scoreList=posInPosPlateScoreList,fileName='posInPosPlateScore')
        save_score_list_to_excel(datasetName=datasetName,datasetVersion=datasetVersion,modelVersion=modelVersion,scoreList=dmsoInPosPlateScoreList,fileName='dmsoInPosPlateScore')

def get_significant_gene(datasetName,datasetVersion,modelVersion):
    if datasetName=='liver':
        inputDim=52
        geneListName='gene'
    elif datasetName=='skin':
        inputDim=1130
        geneListName='GENE'
    model=FC(inputDim=inputDim).to('cuda')
    model.load_state_dict(torch.load('../weight/'+datasetName+'/'+datasetVersion+'/'+modelVersion+'.pth'))
    geneWeight=model.state_dict()['net.0.weight'][0].cpu().numpy()
    plot.bar([i for i in range(inputDim)],geneWeight,width=0.9)
    plot.savefig('../result/'+datasetName+'/'+datasetVersion+'/'+modelVersion+'_gene.png')
    data:dict=pickle.load(open('../data/'+datasetName+'/'+datasetVersion+'/raw.pkl', 'rb'))
    geneName=data[geneListName]
    posWeightList=[]
    negWeightList=[]
    for i in range(inputDim):
        if geneWeight[i]>0:
            posWeightList.append((geneName[i],geneWeight[i]))
        else:
            negWeightList.append((geneName[i],geneWeight[i]))
    posWeightList.sort(key=lambda x:x[1],reverse=True)
    negWeightList.sort(key=lambda x:x[1],reverse=False)
    posDf=pd.DataFrame(posWeightList,columns=['gene','weight'])
    negDf=pd.DataFrame(negWeightList,columns=['gene','weight'])
    posDf.to_excel('../result/'+datasetName+'/'+datasetVersion+'/'+modelVersion+'_posGene.xlsx')
    negDf.to_excel('../result/'+datasetName+'/'+datasetVersion+'/'+modelVersion+'_negGene.xlsx')

if __name__=="__main__":
    # create_dataset(datasetName='liver',datasetVersion='old')
    # create_dataset(datasetName='skin',datasetVersion='new')

    # run_genetic_algorithm(datasetName='liver',datasetVersion='old',modelVersion='3')
    # run_genetic_algorithm(datasetName='skin',datasetVersion='new',modelVersion='1')
    
    draw_scatter(datasetName='liver',datasetVersion='old',modelVersion='3')
    # draw_scatter(datasetName='skin',datasetVersion='new',modelVersion='1')

    # score_all_compound(datasetName='liver',datasetVersion='old',modelVersion='3')
    # score_all_compound(datasetName='skin',datasetVersion='new',modelVersion='1')

    # get_significant_gene(datasetName='liver',datasetVersion='old',modelVersion='3')
    # get_significant_gene(datasetName='skin',datasetVersion='new',modelVersion='1')

    print("All is well!")
    writer.close()
