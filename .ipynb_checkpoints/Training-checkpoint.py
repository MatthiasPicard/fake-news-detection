import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from sklearn.metrics import f1_score

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
            train_acc, test_acc = self.test_f1()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train F1: {train_acc:.4f}, Test F1: {test_acc:.4f}')

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
    

    def test_f1(self):
        self.model.eval()
        pred = self.model(self.data.x_dict, self.data.edge_index_dict).argmax(dim=-1)
    
        f1_scores = []
        for split in ['train_mask', 'test_mask']:
            mask = self.data[self.label][split]
            f1 = f1_score(self.data[self.label].y[mask], pred[mask], average='weighted')
            f1_scores.append(f1)
        return f1_scores

    
