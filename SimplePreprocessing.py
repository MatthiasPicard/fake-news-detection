import numpy as np

from sklearn.model_selection import train_test_split
import pandas as pd
from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T
from Preprocessing import Preprocessing,EMBEDDING_EVENT,IF_NO_EMBEDDING_KEEP

class SimplePreprocessing(Preprocessing):
     
    # TODO, if no embedding, there are probably other stuff to add
    def _define_features_events(self,df):
        df = super(SimplePreprocessing,self)._define_features_events(df)
        df = df.drop(EMBEDDING_EVENT,axis=1)
        df = df.dropna(subset=IF_NO_EMBEDDING_KEEP)
        df = pd.get_dummies(df, columns=IF_NO_EMBEDDING_KEEP)
        return df.astype(float)
    
    # TODO we could probably factorize this code to the main class without too much trouble                
    def create_graph(self,labels,df_events,df_mentions,mode = "train"): 
        
        df_mentions,labels_sorted,mapping_source,y = self._create_label_node(labels,df_mentions)
        df_mentions,mapping_article,df_article_sorted = self._create_non_label_node(df_mentions)
        df_mentions,_, df_events_sorted = self._create_event_node(df_events,df_mentions)
        
        edge_est_source_de,df_mentions = self._create_est_source_de_edge(df_mentions,mapping_article,mapping_source)
        
        edge_mentionné,_ = self._create_mentionne_edge(df_mentions,mapping_article,mapping_source)       

        # NOTE ideally, this function should be applied separately to the train and to the test
        df_events_sorted_temp = self._define_features_events(df_events_sorted) 
        print(df_events_sorted_temp) 
        print(df_events_sorted_temp.columns)
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