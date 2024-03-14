from abc import ABC, abstractmethod
from SimplePreprocessing import SimplePreprocessing
from EventConnexionPreprocessing import EventConnexionPreprocessing
from Heterogemodel import HAN
from Training import SimpleTraining
import torch

class TrainingService(ABC):
    
    def __init__(self,label,device):
        
        self.label = label
        self.device = device
    
    @abstractmethod
    def create_graph_and_train_on_model(self):
        pass
    
    # directly train a model on an already created graph 
    # could be useful to implement if the preprocessing become time expensive if we scale a lot
    def import_graph_and_train_on_model(self):
        pass
    
    
class SimpleConnexionsHAN(TrainingService): 
        
    def create_graph_and_train_on_model(self,list_event,list_mention,hidden_channels,out_channels,n_heads,nb_epoch,lr,weight_decay=0,dropout=None):
        
        preprocessing = SimplePreprocessing(self.label)
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
        
class CloseEventsConnexionsHAN(TrainingService): 
     
    # TODO, make an init to add the attribute "col" and change create_graph accordingly
    # def __init__(self,label,device,col):
    #     super().__init__(label,device)
    #     self.col = col
    
    def create_graph_and_train_on_model(self,list_event,list_mention,hidden_channels,out_channels,n_heads,nb_epoch,lr,weight_decay=0,dropout=None):
        
        preprocessing = EventConnexionPreprocessing(self.label)
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

        