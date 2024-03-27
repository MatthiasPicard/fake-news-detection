from SimplePreprocessing import SimplePreprocessing
from Heterogemodel import HAN
from Training import SimpleTraining
from TrainingService import SimpleConnexionsHAN,CloseEventsConnexionsHAN,EmbeddedFeaturesEventHAN

from GraphViz import GraphViz
import torch
import torch.nn.functional as F
import os

def get_csv_files(directory, n):
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".CSV"):
            # print(os.path.join(directory, filename))
            csv_files.append(os.path.join(directory, filename))
            if len(csv_files) == n:
                break
    return csv_files

if __name__ == "__main__":
    
    nb_event_csv = 1 # TODO: fail at 17 if label = source ( bug fixé mais un peu à la zob)
    nb_mentions_csv = 1
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
    "nb_epoch": 10,
    
    "lr": 0.005,
    "weight_decay":0.001
    
    }
       
    fake_news_detector = SimpleConnexionsHAN(label,device) 
    # fake_news_detector = CloseEventsConnexionsHAN(label,device) 
    # fake_news_detector = EmbeddedFeaturesEventHAN(label,device)
    fake_news_detector.create_graph_and_train_on_model(**args_simple_connexions_HAN_1)
    
    # TODO: create a function to save model
    # TODO faire en sorte que les resultats soient reproductibles
    
    # analyse = GraphViz(label,list_event,list_mention)
    # analyse.get_recap()
    # analyse.display_graph()








