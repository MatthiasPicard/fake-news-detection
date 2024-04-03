from abc import ABC, abstractmethod
from torchinfo import summary
from SimplePreprocessing import SimplePreprocessing
from EventConnexionPreprocessing import EventConnexionPreprocessing
from EmbeddedFeaturesEventPreprocessing import EmbeddedFeaturesEventPreprocessing
from EmbeddedFeaturesEventAndConnexionPreprocessing import EmbeddedFeaturesEventAndConnexionPreprocessing
from Heterogemodel import HAN
from Training import SimpleTraining
from torch_geometric.data import Data, DataLoader
import torch
import os

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
    
    def create_graph_and_save_model(self):
        pass
    
    
class SimpleConnexionsHAN(TrainingService): 
        
    def create_graph_and_train_on_model(self,list_event,list_mention,hidden_channels,out_channels,n_heads,nb_epoch,lr,weight_decay=0,dropout=None):
        
        preprocessing = SimplePreprocessing(self.label,self.is_mixte)
        labels,df_events,df_mentions = preprocessing.data_load(list_event,list_mention)
        data = preprocessing.create_graph(labels,df_events,df_mentions)
        
        model = HAN(self.label,metadata = data.metadata(),
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    n_heads=n_heads,
                    dropout = dropout)
        
        # print(summary(model))
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
        
        
class EmbeddedFeaturesEventAndConnexionstHAN(CloseEventsConnexionsHAN): 
 
    def create_graph_and_train_on_model(self,list_event,list_mention,hidden_channels,out_channels,n_heads,nb_epoch,lr,weight_decay=0,dropout=None):
        
        preprocessing = EmbeddedFeaturesEventAndConnexionPreprocessing(self.label,self.is_mixte,self.col)
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
        
class EmbeddedFeaturesEventHAN(TrainingService): 
 
    def create_graph_and_train_on_model(self,list_event,list_mention,hidden_channels,out_channels,n_heads,nb_epoch,lr,weight_decay=0,dropout=None):
        
        preprocessing = EmbeddedFeaturesEventPreprocessing(self.label,self.is_mixte)
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

        