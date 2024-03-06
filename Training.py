import torch
import torch.nn.functional as F

class Training():
    
    def __init__(self,data,model,optimizer,nb_epoch):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.nb_epoch = nb_epoch
        
    def train(self):
        
        with torch.no_grad():
            out = self.model(self.data.x_dict, self.data.edge_index_dict)
        
        for epoch in range(0, self.nb_epoch):
            loss = self.train_one_epoch()
            train_acc, test_acc = self.test()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')

    def train_one_epoch(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x_dict, self.data.edge_index_dict)
        mask = self.data['source'].train_mask
        # print(mask)
        # print(out)
        # print(self.data['source'].y[mask])
        loss = F.cross_entropy(out[mask], self.data['source'].y[mask].long())
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        pred = self.model(self.data.x_dict, self.data.edge_index_dict).argmax(dim=-1)

        accs = []
        for split in ['train_mask', 'test_mask']:
            mask = self.data['source'][split]
            acc = (pred[mask] == self.data['source'].y[mask]).sum() / mask.sum()
            accs.append(float(acc))
        return accs
    