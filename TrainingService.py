from abc import ABC, abstractmethod
from SimplePreprocessing import SimplePreprocessing
from EventConnexionPreprocessing import EventConnexionPreprocessing
from heterogemodel import HAN
from Training import SimpleTraining
import torch
from torch_geometric.loader import NeighborLoader, ImbalancedSampler
import matplotlib.pyplot as plt

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
        
        
        
        
        
        results = []
        for l in lr:
            for e in nb_epoch:
                model = HAN(self.label,metadata = data.metadata(),
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    n_heads=n_heads,
                    dropout = dropout)
                data, model = data.to(self.device), model.to(self.device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=l, weight_decay=weight_decay)
                optimizer.zero_grad()
                training_process = SimpleTraining(data,model,optimizer,e,self.label)
                result = training_process.train()
                #Results : ((learning rate,epoch,(Recall,Precision,F1 Score)))
                results.append((l, e, result))
        
        for lr, epoch, result in results:
            print(f"Learning Rate: {lr}, Epochs: {epoch}, Result(Precision,Recall,F1 score): {result}")
        
        
        
        
        for e in [5,10,15,50,100,500,1000]:
            model = HAN(self.label,metadata = data.metadata(),
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    n_heads=n_heads,
                    dropout = dropout)
            data, model = data.to(self.device), model.to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=weight_decay)
            #optimizer.zero_grad()
            training_process = SimpleTraining(data,model,optimizer,e,self.label)
            result = training_process.train()
            #Results : ((learning rate,epoch,(Recall,Precision,F1 Score)))
            results.append((l, e, result))
            # Plotting the results
            x = []
            y = []
            for lr, epoch, result in results:
                x.append(epoch)
                y.append(result[2])

        plt.plot(x, y)
        plt.xlabel('Epochs')
        plt.ylabel('Result (F1 Score) avec lr = 0.1')
        plt.title('Training Results')
        plt.show()
        
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

        