from abc import ABC, abstractmethod
import torch_geometric
from torchinfo import summary
from Preprocessing.SimplePreprocessing import SimplePreprocessing
from Preprocessing.EventConnexionPreprocessing import EventConnexionPreprocessing
from Preprocessing.EmbeddedFeaturesEventPreprocessing import EmbeddedFeaturesEventPreprocessing
from Preprocessing.EmbeddedFeaturesEventAndConnexionPreprocessing import EmbeddedFeaturesEventAndConnexionPreprocessing
from Heterogemodel import HAN
from Training import SimpleTraining
from torch_geometric.data import Data, DataListLoader
from torch_geometric.loader import  NeighborLoader
import torch
import os
import numpy as np


SAVED_GRAPHS_DIR = "saved_graphs"

class TrainingService(ABC):
    """Parent class of all classes that encapsulate all the services 
    that the user want to use for a specific preprocessing and training"""
    
    def __init__(self,label,is_mixte,device): 
        self.label = label
        self.is_mixte = is_mixte
        self.device = device
    
    @abstractmethod
    def create_graph_and_train_on_model(self):
        pass
    
    def import_graph_and_train_on_model(self,name,hidden_channels,out_channels,n_heads,nb_epoch,lr,weight_decay=0,dropout=None,batch_size = 16):
        """Directly train a HAN model on an already created graph"""
        
        save_path = os.path.join(SAVED_GRAPHS_DIR, name)
        data = torch.load(save_path)
        model = HAN(self.label,metadata = data.metadata(),
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    n_heads=n_heads,
                    dropout = dropout)
        
        data, model = data.to(self.device), model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_input_nodes = (self.label, data[self.label].train_mask)
        train_loader = NeighborLoader(data, num_neighbors=[10]*2,input_nodes=train_input_nodes,shuffle=True,batch_size=batch_size)
        training_process = SimpleTraining(data,model,optimizer,nb_epoch,self.label)
        training_process.train(train_loader)
    
    def create_graph_and_save(self):
        pass
    
"""All the classes below are a service that uses a different preprocessing,
but always the same model.
"""

   
class SimpleConnexionsHAN(TrainingService): 
        
    def create_graph_and_train_on_model(self,list_event,list_mention,hidden_channels,out_channels,n_heads,nb_epoch,lr,weight_decay=0,dropout=None,batch_size = 16):
        preprocessing = SimplePreprocessing(self.label,self.is_mixte)
        labels,df_events,df_mentions = preprocessing.data_load(list_event,list_mention)
        data = preprocessing.create_graph(labels,df_events,df_mentions)

        model = HAN(self.label,metadata = data.metadata(),
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    n_heads=n_heads,
                    dropout = dropout)
                
        data, model = data.to(self.device), model.to(self.device)        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_input_nodes = (self.label, data[self.label].train_mask)
        train_loader = NeighborLoader(data, num_neighbors=[10]*2,input_nodes=train_input_nodes,shuffle=True,batch_size=batch_size)
        training_process = SimpleTraining(data,model,optimizer,nb_epoch,self.label)             
        training_process.train(train_loader)
   
    def create_graph_and_save(self,list_event,list_mention,name):
        preprocessing = SimplePreprocessing(self.label,self.is_mixte)
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
    
    def create_graph_and_train_on_model(self,list_event,list_mention,hidden_channels,out_channels,n_heads,nb_epoch,lr,weight_decay=0,dropout=None,batch_size = 16):
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
        train_input_nodes = (self.label, data[self.label].train_mask)
        train_loader = NeighborLoader(data, num_neighbors=[10]*2,input_nodes=train_input_nodes,shuffle=True,batch_size=batch_size)
        training_process = SimpleTraining(data,model,optimizer,nb_epoch,self.label)
        training_process.train(train_loader)
    
    def create_graph_and_save(self,list_event,list_mention,name):
        preprocessing = EventConnexionPreprocessing(self.label,self.is_mixte,self.col)
        labels,df_events,df_mentions = preprocessing.data_load(list_event,list_mention)
        data = preprocessing.create_graph(labels,df_events,df_mentions)
        if not os.path.exists(SAVED_GRAPHS_DIR):
            os.makedirs(SAVED_GRAPHS_DIR)
             
        save_path = os.path.join(SAVED_GRAPHS_DIR, name)
        torch.save(data, save_path)           
        
        
class EmbeddedFeaturesEventAndConnexionstHAN(CloseEventsConnexionsHAN): 
 
    def create_graph_and_train_on_model(self,list_event,list_mention,hidden_channels,out_channels,n_heads,nb_epoch,lr,weight_decay=0,dropout=None,batch_size = 16):
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
        train_input_nodes = (self.label, data[self.label].train_mask)
        train_loader = NeighborLoader(data, num_neighbors=[10]*2,input_nodes=train_input_nodes,shuffle=True,batch_size=batch_size)
        training_process = SimpleTraining(data,model,optimizer,nb_epoch,self.label)
        training_process.train(train_loader)
    
    def create_graph_and_save(self,list_event,list_mention,name):
        preprocessing = EmbeddedFeaturesEventAndConnexionPreprocessing(self.label,self.is_mixte,self.col)
        labels,df_events,df_mentions = preprocessing.data_load(list_event,list_mention)
        data = preprocessing.create_graph(labels,df_events,df_mentions)
        if not os.path.exists(SAVED_GRAPHS_DIR):
            os.makedirs(SAVED_GRAPHS_DIR)
             
        save_path = os.path.join(SAVED_GRAPHS_DIR, name)
        torch.save(data, save_path)  
                 
class EmbeddedFeaturesEventHAN(TrainingService): 
 
    def create_graph_and_train_on_model(self,list_event,list_mention,hidden_channels,out_channels,n_heads,nb_epoch,lr,weight_decay=0,dropout=None,batch_size = 16):
        preprocessing = EmbeddedFeaturesEventPreprocessing(self.label,self.is_mixte)
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
        train_loader = torch_geometric.loader.DataLoader([data], batch_size=batch_size)
        training_process = SimpleTraining(data,model,optimizer,nb_epoch,self.label)
        training_process.train(train_loader)
        
    def create_graph_and_save(self,list_event,list_mention,name):
        preprocessing = EmbeddedFeaturesEventPreprocessing(self.label,self.is_mixte)
        labels,df_events,df_mentions = preprocessing.data_load(list_event,list_mention)
        data = preprocessing.create_graph(labels,df_events,df_mentions)
        if not os.path.exists(SAVED_GRAPHS_DIR):
            os.makedirs(SAVED_GRAPHS_DIR)
            
        save_path = os.path.join(SAVED_GRAPHS_DIR, name)
        torch.save(data, save_path) 

                
        


        