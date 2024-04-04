from SimplePreprocessing import SimplePreprocessing
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
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".CSV"):
            # print(os.path.join(directory, filename))
            csv_files.append(os.path.join(directory, filename))
            if len(csv_files) == n:
                break
    return csv_files

if __name__ == "__main__":
    
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    nb_event_csv = 100 # TODO: (672 for a week)
    nb_mentions_csv = 100
    list_mention = get_csv_files("gdelt_data", nb_mentions_csv)
    list_event = get_csv_files("gdelt_data_event",nb_event_csv)

    label = "source"
    is_mixte = True
    device = torch.device('cpu')

    args_simple_connexions_HAN_1 = {
    "list_mention": list_mention,
    "list_event":list_event,
    
    "hidden_channels": 64,
    "out_channels": 2,
    "n_heads": 4,
    "dropout": 0.2,
    "nb_epoch": 30,
    
    "lr": 0.005,
    "weight_decay":0.001
    }
    
    name_save = "100_source_100_event_50_epoch_4_heads_64_hidden_0.2_dropout_0.005_lr_0.001_weight_decay_ismixte_True_embedding_and_connexions_Actor1Name"
    name_load = "100_source_100_event_50_epoch_4_heads_64_hidden_0.2_dropout_0.005_lr_0.001_weight_decay_ismixte_True_embedding_and_connexions_Actor1Name"
    list_arg_save_graph = ["list_mention",'list_event']
    list_arg_load_graph = ["hidden_channels","out_channels","n_heads","dropout","nb_epoch","lr","weight_decay"]
    args_save_graph = {key:args_simple_connexions_HAN_1[key] for key in list_arg_save_graph}
    args_load_graph = {key:args_simple_connexions_HAN_1[key] for key in list_arg_load_graph}

       
    # fake_news_detector = SimpleConnexionsHAN(label,is_mixte,device) 
    # fake_news_detector = CloseEventsConnexionsHAN(label,is_mixte,device,col="Actor1Name") 
    # fake_news_detector = EmbeddedFeaturesEventHAN(label,is_mixte,device)
    fake_news_detector = EmbeddedFeaturesEventAndConnexionstHAN(label,is_mixte,device,col="Actor1Name") 
    # fake_news_detector.create_graph_and_train_on_model(**args_simple_connexions_HAN_1)
    
    # fake_news_detector.create_graph_and_save(**args_save_graph,name = name_save)
    fake_news_detector.import_graph_and_train_on_model(**args_load_graph,name = name_load)


    # TODO tester différents hyperparamètres
    # TODO tester différents models
    # TODO avec des features ça serait sans doute mieux
        
    # analyse = GraphViz(label,list_event,list_mention)
    # analyse.get_recap()
    # analyse.display_graph()








