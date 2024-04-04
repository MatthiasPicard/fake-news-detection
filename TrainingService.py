from abc import ABC, abstractmethod
import torch_geometric
from torchinfo import summary
from SimplePreprocessing import SimplePreprocessing
from EventConnexionPreprocessing import EventConnexionPreprocessing
from EmbeddedFeaturesEventPreprocessing import EmbeddedFeaturesEventPreprocessing
from Heterogemodel import HAN
from Training import SimpleTraining
from torch_geometric.data import Data, DataLoader
import torch
import pickle
import os
import numpy as np
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import HeteroData

SAVED_GRAPHS_DIR = "saved_graphs"

class TrainingService(ABC):
    
    def __init__(self,label,is_mixte,device):
        
        self.label = label
        self.is_mixte = is_mixte
        self.device = device
    
    @abstractmethod
    def create_graph_and_train_on_model(self):
        pass
    
    # directly train a model on an already created graph 
    # could be useful to implement if the preprocessing become time expensive if we scale a lot
    def import_graph_and_train_on_model(self):
        pass
    
    def create_graph_and_save_model(self):
        pass
    
    
class SimpleConnexionsHAN(TrainingService): 
        
    def create_graph_and_train_on_model(self,list_event,list_mention,hidden_channels,out_channels,n_heads,nb_epoch,lr,weight_decay=0,dropout=None):
        
        preprocessing = SimplePreprocessing(self.label,self.is_mixte)
        labels,df_events,df_mentions = preprocessing.data_load(list_event,list_mention)
        data = preprocessing.create_graph(labels,df_events,df_mentions)
        
        results = []
        for l in lr:
            for e in nb_epoch:
                model = HAN(self.label,metadata = data.metadata(),
                            hidden_channels=hidden_channels,
                            out_channels=out_channels,
                            n_heads=n_heads,
                            dropout = dropout)
                
                # print(summary(model))
                
                
                data, model = data.to(self.device), model.to(self.device)
                
                batch_size = 16
                train_dataset = data[:int(0.8*len(data))]
                
                try:
                    target = train_dataset.y.tolist()
                except:
                    target = train_dataset.y
                class_sample_count = np.array(
                    [len(np.where(target == t)[0]) for t in np.unique(target)])
                
                weight = 1. / class_sample_count
                samples_weight = np.array([weight[int(t)] for t in target])
                samples_weight = torch.from_numpy(samples_weight)
                samples_weigth = samples_weight.double()
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            
                train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
                optimizer = torch.optim.Adam(model.parameters(), lr=l, weight_decay=weight_decay)
                training_process = SimpleTraining(data,model,optimizer,e,self.label)             
                result = training_process.train(train_loader)
                results.append((l,e,result))
        
        for lr, epoch, result in results:
            print(f"Learning Rate: {lr}, Epochs: {epoch}, Result(Precision,Recall,F1 score): {result}")
        
class CloseEventsConnexionsHAN(TrainingService): 
     
    def __init__(self,label,is_mixte,device,col="EventCode"):
        super().__init__(label,is_mixte,device)
        self.col = col
    
    def create_graph_and_train_on_model(self,list_event,list_mention,hidden_channels,out_channels,n_heads,nb_epoch,lr,weight_decay=0,dropout=None):
        
        preprocessing = EventConnexionPreprocessing(self.label,self.is_mixte,self.col)
        labels,df_events,df_mentions = preprocessing.data_load(list_event,list_mention)
        data = preprocessing.create_graph(labels,df_events,df_mentions)
        
        model = HAN(self.label,metadata = data.metadata(),
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    n_heads=n_heads,
                    dropout = dropout)
        
        data, model = data.to(self.device), model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        training_process = SimpleTraining(data,model,optimizer,nb_epoch,self.label)
        training_process.train()
    
    def create_graph_and_save(self,list_event,list_mention,name):
        preprocessing = EventConnexionPreprocessing(self.label,self.is_mixte,self.col)
        labels,df_events,df_mentions = preprocessing.data_load(list_event,list_mention)
        data = preprocessing.create_graph(labels,df_events,df_mentions)
        if not os.path.exists(SAVED_GRAPHS_DIR):
            os.makedirs(SAVED_GRAPHS_DIR)
            
        save_path = os.path.join(SAVED_GRAPHS_DIR, name)
        torch.save(data, save_path)
        
    def import_graph_and_train_on_model(self,name,hidden_channels,out_channels,n_heads,nb_epoch,lr,weight_decay=0,dropout=None):
        save_path = os.path.join(SAVED_GRAPHS_DIR, name)
        data = torch.load(save_path)
        model = HAN(self.label,metadata = data.metadata(),
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    n_heads=n_heads,
                    dropout = dropout)
        
        data, model = data.to(self.device), model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        training_process = SimpleTraining(data,model,optimizer,nb_epoch,self.label)
        training_process.train()
        

          
class EmbeddedFeaturesEventHAN(TrainingService): 
    
    """Pytorch Geometric Dataset class for the HAN model."""
    """
    class Dataset(torch_geometric.data.Dataset):
        def __init__(self, data):
            super().__init__()
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]
    """
 
    def create_graph_and_train_on_model(self,list_event,list_mention,hidden_channels,out_channels,n_heads,nb_epoch,lr,weight_decay=0,dropout=None):
        
        preprocessing = EmbeddedFeaturesEventPreprocessing(self.label,self.is_mixte)
        labels,df_events,df_mentions = preprocessing.data_load(list_event,list_mention)
        data = preprocessing.create_graph(labels,df_events,df_mentions)
        
        results = []
        for l in lr:
            for e in nb_epoch:
                class CustomDataset(InMemoryDataset):
                    def __init__(self, data):
                        super(CustomDataset, self).__init__("")
                        self.data = data

                    def _download(self):
                        pass

                    def _process(self):
                        self.data = data
                
                
                batch_size = 16

                train_dataset = data
                #train_dataset = CustomDataset(train_dataset)

                mask = data[self.label]["train_mask"]
                try:
                    target = data[self.label].y[mask].tolist()
                except:
                    target = data[self.label].y[mask]
                class_sample_count = np.array(
                    [len(np.where(target == t)[0]) for t in np.unique(target)])
                
                weight = 1. / class_sample_count
                samples_weight = np.array([weight[int(t)] for t in target])
                samples_weight = torch.from_numpy(samples_weight)
                samples_weigth = samples_weight.double()
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
                
                

                train_loader = torch_geometric.loader.DataLoader([train_dataset], batch_size=batch_size)
                model = HAN(self.label,metadata = data.metadata(),
                            hidden_channels=hidden_channels,
                            out_channels=out_channels,
                            n_heads=n_heads,
                            dropout = dropout)
                
                # print(summary(model))
                data, model = data.to(self.device), model.to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=l, weight_decay=weight_decay)
                training_process = SimpleTraining(data,model,optimizer,e,self.label)
                
                result = training_process.train(train_loader)
                results.append((l,e,result))
                
        
        for lr, epoch, result in results:
            print(f"Learning Rate: {lr}, Epochs: {epoch}, Result(Precision,Recall,F1 score): {result}")

        