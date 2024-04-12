import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

class Training(ABC):
    """Parent class of all training classes that should implement the training process and an evaluation"""
    
    def __init__(self,data,model,optimizer,nb_epoch,label):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.nb_epoch = nb_epoch
        self.label = label
    
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def test(self):
        pass
    
    @torch.no_grad()
    def _final_audit(self):
        """ Get some important metrics at the end of the evaluation """
        
        mask = self.data[self.label].test_mask
        pred = self.model(self.data.x_dict, self.data.edge_index_dict).argmax(dim=-1)
        precision = precision_score(self.data[self.label].y[mask].long(),pred[mask])
        recall = recall_score(self.data[self.label].y[mask].long(),pred[mask])
        f1 = f1_score(self.data[self.label].y[mask].long(),pred[mask])
        return precision, recall, f1


class SimpleTraining(Training):
    """Implement a basic training function with classic metrics for a classification task"""
    
    def train(self,train_loader):
        with torch.no_grad():
            out = self.model(self.data.x_dict, self.data.edge_index_dict)
        
        liste_loss = []
        liste_f1 = []
        liste_precision = []
        liste_recall = []
    
        for epoch in range(0, self.nb_epoch):
            loss = self._train_one_epoch(train_loader)
            liste_loss.append(loss)
            if epoch % 10 == 0:
                train_acc, test_acc,precision,recall,f1 = self.test()
            liste_f1.append(f1)
            liste_precision.append(precision)
            liste_recall.append(recall)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}') 
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(liste_loss)
        axs[0, 0].set_title('Loss')
        axs[0, 1].plot(liste_f1)
        axs[0, 1].set_title('F1 Score')
        axs[1, 0].plot(liste_precision)
        axs[1, 0].set_title('Precision')
        axs[1, 1].plot(liste_recall)
        axs[1, 1].set_title('Recall')
        plt.tight_layout()
        plt.show()
          

    def _train_one_epoch(self,train_loader) -> float:
        print(len(train_loader))
        for batch in tqdm(train_loader):
            # print(batch)
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(batch.x_dict, batch.edge_index_dict)#[self.label][:batch_size]
            mask = batch[self.label].train_mask
            b_counter = Counter(batch[self.label].y[mask].long().detach().cpu().tolist())
            b_weights = torch.tensor([sum(batch[self.label].y[mask].long().detach().cpu().tolist()) / b_counter[label] if b_counter[label] > 0 else 0 for label in range(2)])
            # b_weights = b_weights.to(self.device)
            # print(mask)
            # print(out)
            # print(self.data['source'].y[mask])
            loss_function = nn.CrossEntropyLoss()#weight=b_weights
            loss = loss_function(out[mask], batch[self.label].y[mask].long())
            loss.backward()
            self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(self):
        """Get some metrics on the trainset and the testset"""
        
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
    