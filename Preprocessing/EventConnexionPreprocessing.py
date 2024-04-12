import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T
from itertools import product
from Preprocessing import Preprocessing,EMBEDDING_EVENT,IF_NO_EMBEDDING_KEEP
from torch_geometric.utils import remove_self_loops


class EventConnexionPreprocessing(Preprocessing):
    """create a graph that contains new connections between events """

    def __init__(self,label,is_mixte,col):
        super().__init__(label,is_mixte)
        self.col = col
            
    def _create_same_column_edge(self,col,df_events,event_map):
        """create edges between events that have the same value 'col'
        The function can take quite some time to execute if the number of events is high"""

        same_actions = df_events[["GlobalEventID",col]]
        same_actions = same_actions.dropna(subset=[col])
        edge_same_event = torch.tensor([], dtype=torch.long)
        while len(same_actions) > 0:
            same_action_nodes = (same_actions[col] == same_actions.iloc[0][col])
            edges = list(product(same_actions[same_action_nodes]['GlobalEventID'], repeat=2))
            edges_t = torch.tensor(list(zip(*edges)))
            edge_same_event = torch.cat((edge_same_event, edges_t), dim=1)
            same_actions = same_actions.drop(same_actions[same_action_nodes].index)
            
        df = pd.DataFrame(edge_same_event).transpose()
        df[0] = df[0].map(event_map["index"])
        df[1] = df[1].map(event_map["index"])
        df = df.dropna(axis=0, how='any')
        edge_same_event_0 = list(df[0].astype(int))
        edge_same_event_1 = list(df[1].astype(int))
        edge_same_event = torch.tensor([edge_same_event_0,edge_same_event_1])
        edge_same_event, _ = remove_self_loops(edge_same_event)
        return edge_same_event        
    
    def _define_features_events(self,df):
        """As we do not create embeddings, we remove the CAMEO features from the event features"""
        
        df = super(EventConnexionPreprocessing,self)._define_features_events(df)
        df = df.drop(EMBEDDING_EVENT,axis=1)
        df = df.dropna(subset=IF_NO_EMBEDDING_KEEP)
        df = pd.get_dummies(df, columns=IF_NO_EMBEDDING_KEEP)
        return df.astype(float)
    
    def create_graph(self,labels,df_events,df_mentions,mode = "train"): 
        """Main function to create the graph"""
        
        df_mentions,labels_sorted,mapping_source,y = self._create_label_node(labels,df_mentions)
        df_mentions,mapping_article,df_article_sorted = self._create_non_label_node(df_mentions)
        df_mentions,df_events, df_events_sorted = self._create_event_node(df_events,df_mentions)
        
        edge_est_source_de,df_mentions = self._create_est_source_de_edge(df_mentions,mapping_article,mapping_source)
    
        edge_mentionné,event_map = self._create_mentionne_edge(df_mentions,mapping_article,mapping_source)
        
        edge_same_event = self._create_same_column_edge(self.col,df_events,event_map)
        
        # NOTE ideally, this function should be applied separately to the train and to the test
        df_events_sorted_temp = self._define_features_events(df_events_sorted) 
                
        labels_sorted_temp = labels_sorted.copy()
        labels_sorted_temp["y"] = y
        data = HeteroData()
        if self.label == "source":
            data['article'].x = torch.from_numpy(df_article_sorted.to_numpy()).to(dtype=torch.float32)
            data['source'].x = torch.from_numpy(labels_sorted_temp.to_numpy()).to(dtype=torch.float32)
        elif self.label == "article":
            data['source'].x = torch.from_numpy(df_article_sorted.to_numpy()).to(dtype=torch.float32)
            data['article'].x = torch.from_numpy(labels_sorted_temp.to_numpy()).to(dtype=torch.float32)
        
        data['event'].x = torch.from_numpy(df_events_sorted_temp.to_numpy()).to(dtype=torch.float32)

        data['event', 'mentionne', 'article'].edge_index = torch.from_numpy(edge_mentionné).to(dtype=torch.long)
        data['source', 'est_source_de', 'article'].edge_index = torch.from_numpy(edge_est_source_de).to(dtype=torch.long)
        data['event','evenement_proche','event'].edge_index = edge_same_event.to(dtype=torch.long)
        
        transform = T.RemoveIsolatedNodes()
        data = transform(data)
        data[self.label].y = pd.Series(data[self.label].x[:,1].numpy())
        data[self.label].x = np.delete(data[self.label].x, 1, axis=1)
        data_undirected = T.ToUndirected()(data)
        
        # Create the test and train masks
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
        
        if mode == "analyse": # Used for the GraphViz class
            if self.label == "source":
                return data_undirected,edge_same_event,df_article_sorted,labels_sorted,df_events_sorted_temp,edge_mentionné,edge_est_source_de,y
            if self.label == "article":
                return data_undirected,edge_same_event,labels_sorted,df_article_sorted,df_events_sorted_temp,edge_mentionné,edge_est_source_de,y
        else:
            return data_undirected