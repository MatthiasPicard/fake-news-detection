import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
import torch.nn as nn
import matplotlib.pyplot as plt

class Training(ABC):
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def test(self):
        pass
    
    @torch.no_grad()
    def _final_audit(self):
        
        
        
        mask = self.data[self.label].test_mask
        pred = self.model(self.data.x_dict, self.data.edge_index_dict).argmax(dim=-1)
        precision = precision_score(self.data[self.label].y[mask].long(),pred[mask])
        recall = recall_score(self.data[self.label].y[mask].long(),pred[mask])
        f1 = f1_score(self.data[self.label].y[mask].long(),pred[mask])
        
        """print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)"""
        return precision, recall, f1


class SimpleTraining(Training):
    
    def __init__(self,data,model,optimizer,nb_epoch,label):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.nb_epoch = nb_epoch
        self.label = label
        
    def train(self,train_loader):

        with torch.no_grad():
            out = self.model(self.data.x_dict, self.data.edge_index_dict)
        
        liste_loss = []
        liste_f1 = []
        liste_precision = []
        liste_recall = []
        
        for epoch in range(0, self.nb_epoch):
            loss = self._train_one_epoch()
            liste_loss.append(loss)
            loss = self._train_one_epoch(train_loader)
            train_acc, test_acc = self.test()
            #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')   
        return self._final_audit()    

    def _train_one_epoch(self,train_loader) -> float:
        
        for batch in train_loader:
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(batch.x_dict, batch.edge_index_dict)
            mask = batch[self.label].train_mask
            b_counter = Counter(batch[self.label].y[mask].long().detach().cpu().tolist())
            b_weights = torch.tensor([sum(batch[self.label].y[mask].long().detach().cpu().tolist()) / b_counter[label] if b_counter[label] > 0 else 0 for label in range(2)])
            # b_weights = b_weights.to(self.device)
            # print(mask)
            # print(out)
            # print(self.data['source'].y[mask])
            loss_function = nn.CrossEntropyLoss(weight=b_weights)
            loss = loss_function(out[mask], batch[self.label].y[mask].long())
            loss.backward()
            self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        pred = self.model(self.data.x_dict, self.data.edge_index_dict).argmax(dim=-1)

        accs = []
        for split in ['train_mask', 'test_mask']:
            mask = self.data[self.label][split]
            acc = (pred[mask] == self.data[self.label].y[mask]).sum() / mask.sum()
            accs.append(float(acc))
        return accs
    