import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from sklearn.metrics import precision_score, recall_score, f1_score


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
        
        mask = self.data[self.label].train_mask
        pred = self.model(self.data.x_dict, self.data.edge_index_dict).argmax(dim=-1)
        precision = precision_score(pred[mask], self.data[self.label].y[mask].long())
        recall = recall_score(pred[mask], self.data[self.label].y[mask].long())
        f1 = f1_score(pred[mask], self.data[self.label].y[mask].long())

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
        
        for epoch in range(0, self.nb_epoch):
            loss = self._train_one_epoch()
            train_acc, test_acc = self.test()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')
            
        self._final_audit()    

    def _train_one_epoch(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x_dict, self.data.edge_index_dict)
        mask = self.data[self.label].train_mask
        # print(mask)
        # print(out)
        # print(self.data['source'].y[mask])
        loss = F.cross_entropy(out[mask], self.data[self.label].y[mask].long())
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
    
    
