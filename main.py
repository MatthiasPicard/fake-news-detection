from Preprocessing import Preprocessing
from Heterogemodel import HAN
from Training import Training
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    
    label = "source"
    list_mention = ["gdelt_data/20231001000000.mentions.CSV","gdelt_data/20231001001500.mentions.CSV"]
    list_event = ["gdelt_data_event/20231001000000.export.CSV","gdelt_data_event/20231001001500.export.CSV"]
    
    hidden_channels = 64
    out_channels = 2
    n_heads = 4
    dropout = 0.5
    nb_epoch = 5
    
    lr = 0.005
    weight_decay = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    preprocessing = Preprocessing(label)
    labels,df_events,df_mentions = preprocessing.data_load(list_event,list_mention)
    data = preprocessing.create_graph(labels,df_events,df_mentions)
    
    model = HAN(label,metadata = data.metadata(),
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                n_heads=n_heads,
                dropout = dropout)
    
    data, model = data.to(device), model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    training_process = Training(data,model,optimizer,nb_epoch,label)
    training_process.train()
    
    # TODO: create a function to save model










