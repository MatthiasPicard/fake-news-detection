from Preprocessing.SimplePreprocessing import SimplePreprocessing
from Heterogemodel import HAN
from Training import SimpleTraining
from TrainingService import SimpleConnexionsHAN,CloseEventsConnexionsHAN,EmbeddedFeaturesEventHAN,EmbeddedFeaturesEventAndConnexionstHAN
from GraphViz import GraphViz
import torch
import torch.nn.functional as F
import os
import numpy as np
import random


def get_csv_files(directory, n):
    """get all the csv files name in a repository
    Note that you should first download the data from the GDELT website
    with the code provided in import.py

    Args:
        directory (_type_): directory path
        n (_type_): number of csv files to get

    """
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".CSV"):
            csv_files.append(os.path.join(directory, filename))
            if len(csv_files) == n:
                break
    return csv_files

if __name__ == "__main__":
    
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    nb_event_csv = 35  # 1 = 15 min of data
    nb_mentions_csv = 35
    list_mention = get_csv_files("gdelt_data", nb_mentions_csv)
    list_event = get_csv_files("gdelt_data_event",nb_event_csv)

    label = "article" # what nodes will be labeled "article" or "event"
    is_mixte = False # either to consider MBFCMixed as unreliable or not
    device = torch.device('cpu')

    args_simple_connexions_HAN_1= {
    "list_mention": list_mention,
    "list_event":list_event,
    
    # for training an HAN model
    "hidden_channels": 64, 
    "out_channels": 2,
    "n_heads": 4,
    "dropout": 0.2,
    "nb_epoch": 300,
    "batch_size": 256,
    "lr": 0.005,
    "weight_decay":0.001
    }
    
    name_save = "test"
    name_load = "test"
    list_arg_save_graph = ["list_mention",'list_event']
    list_arg_load_graph = ["hidden_channels","out_channels","n_heads","dropout","nb_epoch","lr","weight_decay","batch_size"]
    args_save_graph = {key:args_simple_connexions_HAN_1[key] for key in list_arg_save_graph}
    args_load_graph = {key:args_simple_connexions_HAN_1[key] for key in list_arg_load_graph}


    """You can uncomment the line of code corresponding to the graphs you want to create and train on"""
      
    # fake_news_detector = SimpleConnexionsHAN(label,is_mixte,device) 
    fake_news_detector = CloseEventsConnexionsHAN(label,is_mixte,device,col="Actor1Name") # create new connexions between event nodes that have the same Actor1Name
    # fake_news_detector = EmbeddedFeaturesEventHAN(label,is_mixte,device)
    # fake_news_detector = EmbeddedFeaturesEventAndConnexionstHAN(label,is_mixte,device,col="Actor1Name") 
    
    """ save a graph, train on a graph, or create on the fly and train """
    
    fake_news_detector.create_graph_and_train_on_model(**args_simple_connexions_HAN_1)
    # fake_news_detector.create_graph_and_save(**args_save_graph,name = name_save)
    # fake_news_detector.import_graph_and_train_on_model(**args_load_graph,name = name_load)
        
    """ experimental: visualize the graph """
        
    # analyse = GraphViz(label,list_event,list_mention,is_mixte)
    # analyse.get_recap()
    # analyse.display_graph()








