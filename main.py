from Preprocessing import SimplePreprocessing
from Heterogemodel import HAN
from Training import SimpleTraining
from TrainingService import SimpleConnexionsHAN
from Serialization import Serialization
from GraphViz import GraphViz
import torch
import torch.nn.functional as F
import os

def get_csv_files(directory, n):
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".CSV"):
            csv_files.append(os.path.join(directory, filename))
            if len(csv_files) == n:
                break
    return csv_files

if __name__ == "__main__":
    
    nb_event_csv = 16 # TODO: fail at 17 if label = source
    nb_mentions_csv = 20
    list_mention = get_csv_files("gdelt_data", nb_mentions_csv)
    list_event = get_csv_files("gdelt_data_event",nb_event_csv)

    label = "source"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args_simple_connexions_HAN_1 = {
        "list_mention": list_mention,
        "list_event":list_event,
    
        "hidden_channels": 64,
        "out_channels": 2,
        "n_heads": 4,
        "dropout": 0.5,
        "nb_epoch": 5,
    
        "lr": 0.005,
        "weight_decay":0.001
    }

    args_simple_connexions_HAN_1_save = {
        "graph_name": "graph.pkl",
        "list_mention": list_mention,
        "list_event":list_event,
    }

    args_simple_connexions_HAN_1_import = {
        "graph_name": "graph.pkl",
    
        "hidden_channels": 64,
        "out_channels": 2,
        "n_heads": 4,
        "dropout": 0.5,
        "nb_epoch": 5,
    
        "lr": 0.005,
        "weight_decay":0.001
    }
       
    fake_news_detector = SimpleConnexionsHAN(label,device) 
    
    # Create and save graph
    # fake_news_detector.create_graph_and_save(**args_simple_connexions_HAN_1_save)

    # Create graph and train on model
    fake_news_detector.create_graph_and_train_on_model(**args_simple_connexions_HAN_1)

    # Import graph and train on model
    # fake_news_detector.import_graph_and_train_on_model(**args_simple_connexions_HAN_1_import)
    
    # analyse = GraphViz(label,list_event,list_mention)
    # analyse.get_recap()








