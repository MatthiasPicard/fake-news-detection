import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T
from itertools import product
from Preprocessing import Preprocessing,EMBEDDING_EVENT,IF_NO_EMBEDDING_KEEP
from sentence_transformers import SentenceTransformer
from EventConnexionPreprocessing import EventConnexionPreprocessing
from EmbeddedFeaturesEventPreprocessing import EmbeddedFeaturesEventPreprocessing
from torch_geometric.utils import remove_self_loops


class EmbeddedFeaturesEventAndConnexionPreprocessing(Preprocessing):
    
    def __init__(self,label,is_mixte,col):
        super().__init__(label,is_mixte)
        self.col = col
    
    def _create_same_column_edge(self,col,df_events,event_map):
        
        same_actions = df_events[["GlobalEventID",col]]
        same_actions = same_actions.dropna(subset=[col])
        edge_same_event = torch.tensor([], dtype=torch.long)
        while len(same_actions) > 0:
            print(len(same_actions))
            same_action_nodes = (same_actions[col] == same_actions.iloc[0][col])#.nonzero().squeeze()
            cartesian_product = list(product(same_actions[same_action_nodes]['GlobalEventID'], repeat=2))
            # edges = [[x, y] for x, y in cartesian_product if x < y]
            edges = cartesian_product
            print('on est la')
            edges_t = torch.tensor(list(zip(*edges)))
            edge_same_event = torch.cat((edge_same_event, edges_t), dim=1)
            same_actions = same_actions.drop(same_actions[same_action_nodes].index)
        df = pd.DataFrame(edge_same_event).transpose()
        df[0] = df[0].map(event_map["index"])
        df[1] = df[1].map(event_map["index"])
        df = df.dropna(axis=0, how='any')
        # print(df)
        # print(df.isna().sum())
        # print(pd.DataFrame(edge_same_event)[0].map(event_map["index"]))
        # print(pd.DataFrame(edge_same_event)[1].map(event_map["index"]))
        edge_same_event_0 = list(df[0].astype(int))
        edge_same_event_1 = list(df[1].astype(int))
        # print(edge_same_event_1)
        edge_same_event = torch.tensor([edge_same_event_0,edge_same_event_1])
        edge_same_event, _ = remove_self_loops(edge_same_event)
        print(len(edge_same_event[0]))
        # print(edge_same_event)
        return edge_same_event
         
    def _define_features_events(self,df):
        df = super(EmbeddedFeaturesEventAndConnexionPreprocessing,self)._define_features_events(df)
        print(df.columns)
        df = df.drop(IF_NO_EMBEDDING_KEEP, axis=1)
        df = self._define_embedding_event(df)
        return df
    
    def _define_embedding_event(self,df):
        
        df["sentence"] = ""
        mapping_event_code = pd.read_csv("cameo.csv",sep=";",names = ["Code","FullName"],dtype="str")
        df["EventCode"] = df['EventCode'].astype(str).map(mapping_event_code.set_index('Code')['FullName'])
        
        is_nan_row = df[EMBEDDING_EVENT].isnull().all(axis=1) 
        df.loc[is_nan_row,"sentence"] = "no information on this event"
        
        embedding_event = set(["EventCode","ActionGeo_Fullname","ActionGeo_CountryCode"])
        embedding_no_event = list(set(EMBEDDING_EVENT)- embedding_event)
        only_event = list(df[embedding_no_event].isnull().all(axis=1)) and list(~df[list(embedding_event)].isnull().all(axis=1))
        only_event_sentence = lambda x: f"{x['EventCode']} happened in {x['ActionGeo_Fullname']}, {x['ActionGeo_CountryCode']}"
        df.loc[only_event,"sentence"] = df.apply(only_event_sentence, axis=1)

        embedding_actor1_and_event = set(["Actor1Name","Actor1CountryCode","EventCode","ActionGeo_Fullname","ActionGeo_CountryCode","Actor1Geo_Fullname","Actor1Geo_CountryCode"])
        embedding_noactor1_and_noevent = list(set(EMBEDDING_EVENT)-embedding_actor1_and_event)
        no_actor2 = list(df[embedding_noactor1_and_noevent].isnull().all(axis=1)) and list(~df[list(embedding_actor1_and_event)].isnull().all(axis=1)) 
        actor1_and_event_sentence = lambda x:f"{x['Actor1Name']},({x['Actor1CountryCode']}) is in {x['Actor1Geo_Fullname']},({x['Actor1Geo_CountryCode']}) and {x['EventCode']} in {x['ActionGeo_Fullname']}, {x['ActionGeo_CountryCode']}"
        df.loc[no_actor2,"sentence"] = df.apply(actor1_and_event_sentence, axis=1)

        embedding_full = ["Actor1Name","EventCode","Actor2Name"]
        full =  list(~df[embedding_full].isnull().any(axis=1))
        full_sentence = lambda x:f"{x['Actor1Name']},({x['Actor1CountryCode']}) is in {x['Actor1Geo_Fullname']},({x['Actor1Geo_CountryCode']}) and {x['EventCode']} in {x['ActionGeo_Fullname']}, {x['ActionGeo_CountryCode']} to {x['Actor2Name']},({x['Actor2CountryCode']}) which is in {x['Actor1Geo_Fullname']},({x['Actor1Geo_CountryCode']})"
        df.loc[full,"sentence"] = df.apply(full_sentence, axis=1)
        
        df = df.drop(EMBEDDING_EVENT,axis=1)
               
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # print("embedding started")
        embeddings = model.encode(df['sentence'])
        # print(embeddings.shape)
        df = pd.concat([df,pd.DataFrame(embeddings)],axis = 1)
        df = df.drop(['sentence'],axis=1)
        # print(df)
        # print(df.columns)
        return df.astype(float)
                    
    def create_graph(self,labels,df_events,df_mentions,mode = "train"): 
        
        df_mentions,labels_sorted,mapping_source,y = self._create_label_node(labels,df_mentions)
        df_mentions,mapping_article,df_article_sorted = self._create_non_label_node(df_mentions)
        df_mentions,_, df_events_sorted = self._create_event_node(df_events,df_mentions)
        
        edge_est_source_de,df_mentions = self._create_est_source_de_edge(df_mentions,mapping_article,mapping_source)
        
        edge_mentionné,event_map = self._create_mentionne_edge(df_mentions,mapping_article,mapping_source)  
             
        edge_same_event = self._create_same_column_edge(self.col,df_events,event_map)
        # NOTE idealy, this function should be applied separately to the train and to the test
        df_events_sorted_temp = self._define_features_events(df_events_sorted) 
        # print(df_events_sorted_temp)
        
        labels_sorted_temp = labels_sorted.copy()
        labels_sorted_temp["y"] = y

        data = HeteroData()
        if self.label == "source":
            data['article'].x = torch.from_numpy(df_article_sorted.to_numpy()).to(dtype=torch.float32)
            data['source'].x = torch.from_numpy(labels_sorted_temp.to_numpy()).to(dtype=torch.float32)
        elif self.label == "article":
            data['source'].x = torch.from_numpy(df_article_sorted.to_numpy()).to(dtype=torch.float32)
            data['article'].x = torch.from_numpy(labels_sorted_temp.to_numpy()).to(dtype=torch.float32)
        
        # data['source'].y = y
        data['event'].x = torch.from_numpy(df_events_sorted_temp.to_numpy()).to(dtype=torch.float32)

        data['event', 'mentionne', 'article'].edge_index = torch.from_numpy(edge_mentionné).to(dtype=torch.long)
        data['source', 'est_source_de', 'article'].edge_index = torch.from_numpy(edge_est_source_de).to(dtype=torch.long)
        data['event','evenement_proche','event'].edge_index = edge_same_event.to(dtype=torch.long)
        
        transform = T.RemoveIsolatedNodes()
        data = transform(data)

        data[self.label].y = pd.Series(data[self.label].x[:,1].numpy())
        data[self.label].x = np.delete(data[self.label].x, 1, axis=1)

        data_undirected = T.ToUndirected()(data)
        # data_undirected = data
        num_labels = len(data_undirected[self.label].y)
        known_indices = np.where(~data_undirected[self.label].y.isna())[0]
        known_labels = data_undirected[self.label].y[known_indices]
        train_labels, test_labels, train_idx, test_idx = train_test_split(known_labels, known_indices, test_size=0.2, random_state=42)
        train_mask = torch.zeros(num_labels, dtype=torch.bool)
        test_mask = torch.zeros(num_labels, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        data_undirected[self.label].train_mask = train_mask
        data_undirected[self.label].test_mask = test_mask
        data_undirected[self.label].y = torch.from_numpy(data_undirected[self.label].y.to_numpy())
        
        if mode == "analyse":
            if self.label == "source":
                return data_undirected,df_article_sorted,labels_sorted,df_events_sorted_temp,edge_mentionné,edge_est_source_de,y
            if self.label == "article":
                return data_undirected,labels_sorted,df_article_sorted,df_events_sorted_temp,edge_mentionné,edge_est_source_de,y
        else:
            return data_undirected