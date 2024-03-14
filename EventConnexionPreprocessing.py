import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T
from itertools import product
from Preprocessing import Preprocessing

class EventConnexionPreprocessing(Preprocessing):
                     
    def _create_same_column_edge(self,col,df_events,event_map):
        
        same_actions = df_events[["GlobalEventID",col]]
        edge_same_event = torch.tensor([], dtype=torch.long)
        while len(same_actions) > 0:
            print(len(same_actions))
            same_action_nodes = (same_actions[col] == same_actions.iloc[0][col])#.nonzero().squeeze()
            cartesian_product = list(product(same_actions[same_action_nodes]['GlobalEventID'], repeat=2))
            edges = [[x, y] for x, y in cartesian_product if x != y and cartesian_product.index((x, y)) < cartesian_product.index((y, x))]
            edges_t = torch.tensor(list(zip(*edges)))
            edge_same_event = torch.cat((edge_same_event, edges_t), dim=1)
            same_actions = same_actions.drop(same_actions[same_action_nodes].index)
        edge_same_event_0 = list(pd.DataFrame(edge_same_event).loc[0].map(event_map["index"]).astype(int))
        edge_same_event_1 = list(pd.DataFrame(edge_same_event).loc[1].map(event_map["index"]).astype(int))
        # print(edge_same_event_1)
        edge_same_event = torch.tensor([edge_same_event_0,edge_same_event_1])
        # print(edge_same_event)

        return edge_same_event
    
    
    def create_graph(self,labels,df_events,df_mentions,col = "EventCode",mode = "train"): 
        
        df_mentions,labels_sorted,mapping_source,y = self._create_label_node(labels,df_mentions)
        df_mentions,mapping_article,df_article_sorted = self._create_non_label_node(df_mentions)
        df_mentions,df_events, df_events_sorted = self._create_event_node(df_events,df_mentions)
        
        edge_est_source_de,df_mentions = self._create_est_source_de_edge(df_mentions,mapping_article,mapping_source)
    
        edge_mentionné,event_map = self._create_mentionne_edge(df_mentions,mapping_article,mapping_source)
        
        edge_same_event = self._create_same_column_edge(col,df_events,event_map)
        
        # Create attributes for the mentionné edges
        # df_mentions_edges = df_mentions.drop(["GlobalEventID","MentionIdentifier","MentionSourceName"], axis = 1)

        # temporarily remove almost all columns for simplicity
        # NOTE we will add the fonction _define_features here
        df_events_sorted_temp = df_events_sorted[["Day"]] 
        # df_mentions_edges_temp = df_mentions_edges[["EventTimeDate"]]
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

        # data['event', 'mentionne', 'article'].edge_attr = torch.from_numpy(df_mentions_edges_temp.to_numpy())

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
                return data_undirected,edge_same_event,df_article_sorted,labels_sorted,df_events_sorted_temp,edge_mentionné,edge_est_source_de,y
            if self.label == "article":
                return data_undirected,edge_same_event,labels_sorted,df_article_sorted,df_events_sorted_temp,edge_mentionné,edge_est_source_de,y
        else:
            return data_undirected