from SimplePreprocessing import SimplePreprocessing
from Heterogemodel import HAN
from Training import SimpleTraining
from TrainingService import SimpleConnexionsHAN,CloseEventsConnexionsHAN,EmbeddedFeaturesEventHAN
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
    
    nb_event_csv = 100 # TODO: fail at 17 if label = source ( bug fixé mais un peu à la zob)
    nb_mentions_csv = 100
    list_mention = get_csv_files("gdelt_data", nb_mentions_csv)
    list_event = get_csv_files("gdelt_data_event",nb_event_csv)

    label = "source"
    is_mixte = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args_simple_connexions_HAN_1 = {
    "list_mention": list_mention,
    "list_event":list_event,
    
    "hidden_channels": 64,
    "out_channels": 2,
    "n_heads": 4,
    "dropout": 0.2,
    "nb_epoch": 10,
    
    "lr": 0.005,
    "weight_decay":0.001
    }
    
    name_save = "test"
    name_load = "test"
    list_arg_save_graph = ["list_mention",'list_event']
    list_arg_load_graph = ["hidden_channels","out_channels","n_heads","dropout","nb_epoch","lr","weight_decay"]
    args_save_graph = {key:args_simple_connexions_HAN_1[key] for key in list_arg_save_graph}
    args_load_graph = {key:args_simple_connexions_HAN_1[key] for key in list_arg_load_graph}

       
    fake_news_detector = SimpleConnexionsHAN(label,is_mixte,device) 
    # fake_news_detector = CloseEventsConnexionsHAN(label,is_mixte,device,col="Actor1Name") 
    # fake_news_detector = EmbeddedFeaturesEventHAN(label,is_mixte,device)
    fake_news_detector.create_graph_and_train_on_model(**args_simple_connexions_HAN_1)
    # fake_news_detector.create_graph_and_save(**args_save_graph,name = name_save)
    # fake_news_detector.import_graph_and_train_on_model(**args_load_graph,name = name_load)


    # TODO créer plus d'analyse pour le training (plot loss and val loss)
    # TODO tester différents hyperparamètres
    # TODO tester différents models( est ce que celui la n'est pas trop gros?)
    # TODO avec des features ça serait sans doute mieux
    # TODO tester ce qu'il se passerait si on mettait les mixed comme des fakes news pour rééquilibrer
    
    # TODO find a way to reduce the time needed to create new connections (this is better now but still not perfect)
    
    # analyse = GraphViz(label,list_event,list_mention)
    # analyse.get_recap()
    # analyse.display_graph()








