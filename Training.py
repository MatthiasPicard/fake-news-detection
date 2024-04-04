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
        print(pred[mask])
        print(self.data[self.label].y[mask].long())
        precision = precision_score(self.data[self.label].y[mask].long(),pred[mask])
        recall = recall_score(self.data[self.label].y[mask].long(),pred[mask])
        f1 = f1_score(self.data[self.label].y[mask].long(),pred[mask])

        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)


class SimpleTraining(Training):
    
    def __init__(self,data,model,optimizer,nb_epoch,label):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.nb_epoch = nb_epoch
        self.label = label
        
    def train(self):

        with torch.no_grad():
            out = self.model(self.data.x_dict, self.data.edge_index_dict)
        
        liste_loss = []
        liste_f1 = []
        liste_precision = []
        liste_recall = []
        
        for epoch in range(0, self.nb_epoch):
            loss = self._train_one_epoch()
            liste_loss.append(loss)
            train_acc, test_acc,precision,recall,f1 = self.test()
            liste_precision.append(precision)
            liste_recall.append(recall)
            liste_f1.append(f1) 
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')
            
        self._final_audit()
         
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 2, 1)
        plt.plot(liste_loss)
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(2, 2, 2)
        plt.plot(liste_precision)
        plt.title('Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')

        plt.subplot(2, 2, 3)
        plt.plot(liste_recall)
        plt.title('Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')

        plt.subplot(2, 2, 4)
        plt.plot(liste_f1)
        plt.title('F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')

        plt.tight_layout()
        plt.show()

    def _train_one_epoch(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x_dict, self.data.edge_index_dict)
        mask = self.data[self.label].train_mask
        b_counter = Counter(self.data[self.label].y[mask].long().detach().cpu().tolist())
        print(b_counter)
        b_weights = torch.tensor([sum(self.data[self.label].y[mask].long().detach().cpu().tolist()) / b_counter[label] if b_counter[label] > 0 else 0 for label in range(2)])
        print(b_weights)
        # b_weights = b_weights.to(self.device)
        # print(mask)
        # print(out)
        # print(self.data['source'].y[mask])
        loss_function = nn.CrossEntropyLoss(weight=b_weights)
        loss = loss_function(out[mask], self.data[self.label].y[mask].long())
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
        
        precision = precision_score(self.data[self.label].y[mask].long(),pred[mask])
        recall = recall_score(self.data[self.label].y[mask].long(),pred[mask])
        f1 = f1_score(self.data[self.label].y[mask].long(),pred[mask])
        return accs[0], accs[1],precision,recall,f1
    
    
