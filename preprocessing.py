import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from torch_geometric.data import HeteroData
import torch
import pickle
import torch_geometric.transforms as T


COL_NAMES_MENTIONS = ["GlobalEventID", "EventTimeDate", "MentionTimeDate", "MentionType",
                          "MentionSourceName", "MentionIdentifier", "SentenceID", "Actor1CharOffset", "Actor2CharOffset",
                          "ActionCharOffset", "InRawText", "Confidence", "MentionDocLen", "MentionDocTone",
                          "SRCLC", "ENG"]  # mention columns

COL_NAMES_EVENTS = ["GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
                    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode", "Actor1EthnicCode",
                    "Actor1Religion1Code", "Actor1Religion2Code", "Actor1Type1Code", "Actor1Type2Code",
                    "Actor1Type3Code",
                    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode", "Actor2EthnicCode",
                    "Actor2Religion1Code", "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code",
                    "Actor2Type3Code",
                    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
                    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
                    "Actor1Geo_Type", "Actor1Geo_Fullname", "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code",
                    "Actor1Geo_ADM2Code",
                    "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID", "Actor2Geo_Type", "Actor2Geo_Fullname",
                    "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat",
                    "Actor2Geo_Long",
                    "Actor2Geo_FeatureID", "ActionGeo_Type", "ActionGeo_Fullname", "ActionGeo_CountryCode",
                    "ActionGeo_ADM1Code",
                    "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID", "DATEADDED",
                    "SOURCEURL"]  # event columns


class Preprocessing():
    def __init__(self,label):
        self.label = label
    
    def data_load(self,list_event,list_mention):
    
        if self.label == "source":
            with open("mixt_labeled_sources.pkl", 'rb') as file:  # label of sources (nan if not present in MBFS)
                labels = pickle.load(file)
        elif self.label == "article":
            with open("mixt_labeled_articles.pkl", 'rb') as file:  # label of articles (nan if not present in MBFS)
                labels = pickle.load(file)
        else:
            raise ValueError("label should be either 'source' or 'article'")

        events_dfs = []
        for file in list_event:
            df = pd.read_csv(file, delimiter='\t', names=COL_NAMES_EVENTS)
            events_dfs.append(df)
        df_events = pd.concat(events_dfs, ignore_index=True)
            
        mentions_dfs = []
        for file in list_mention:
            df = pd.read_csv(file, delimiter='\t', names=COL_NAMES_MENTIONS)
            mentions_dfs.append(df)
        df_mentions = pd.concat(mentions_dfs, ignore_index=True)
            
        return labels,df_events,df_mentions
    
    def _define_features(self): # TODO: function to retrieve features for events,sources
        pass
    
    # TODO: 
    def create_graph(self,labels,df_events,df_mentions,mode = "train"): 
        
        label_encoder_source = LabelEncoder()
        labels['links'] = label_encoder_source.fit_transform(labels['links'])
        label_mapping_source = dict(zip(label_encoder_source.classes_, label_encoder_source.transform(label_encoder_source.classes_)))
        labels_sorted = labels.sort_values(by="links").set_index("links")
        labels_sorted = labels_sorted.reset_index(drop=False)
        mapping_source = labels_sorted["links"]
        labels_sorted["random"] = np.random.rand(len(labels_sorted))
        y = labels_sorted["is_fake"]#.apply(lambda x:int(x) if not pd.isna(x) else x )
        labels_sorted = labels_sorted[["random"]]

        if self.label == "source":
            df_article = pd.DataFrame(df_mentions["MentionIdentifier"])
            label_encoder_article = LabelEncoder()
            df_article['MentionIdentifier'] = label_encoder_article.fit_transform(df_article['MentionIdentifier'])
            label_mapping_article = dict(zip(label_encoder_article.classes_, label_encoder_article.transform(label_encoder_article.classes_)))
            df_article_sorted = df_article.sort_values(by="MentionIdentifier").set_index("MentionIdentifier")
            df_article_sorted = df_article_sorted.reset_index(drop=False)
            mapping_article = df_article_sorted["MentionIdentifier"]
            df_article_sorted["random"] = np.random.rand(len(df_article_sorted))
            df_article_sorted = df_article_sorted[["random"]]
            
        elif self.label == "article":
            df_article = pd.DataFrame(df_mentions["MentionSourceName"])
            label_encoder_article = LabelEncoder()
            df_article["MentionSourceName"] = label_encoder_article.fit_transform(df_article["MentionSourceName"])
            label_mapping_article = dict(zip(label_encoder_article.classes_, label_encoder_article.transform(label_encoder_article.classes_)))
            df_article_sorted = df_article.sort_values(by="MentionSourceName").set_index("MentionSourceName")
            df_article_sorted = df_article_sorted.reset_index(drop=False)
            mapping_article = df_article_sorted["MentionSourceName"]
            df_article_sorted["random"] = np.random.rand(len(df_article_sorted))
            df_article_sorted = df_article_sorted[["random"]]
            
        else:
            raise ValueError("label should be either 'source' or 'article'")

        label_encoder_event = LabelEncoder()
        df_events['GlobalEventID'] = label_encoder_event.fit_transform(df_events['GlobalEventID'])
        label_mapping_event = dict(zip(label_encoder_event.classes_, label_encoder_event.transform(label_encoder_event.classes_)))
        # df_events['GlobalEventID'] = df_events['GlobalEventID'].map(label_mapping_event)
        df_events_sorted = df_events.sort_values(by="GlobalEventID").set_index("GlobalEventID")
        df_events_sorted = df_events_sorted.reset_index(drop=False)
        mapping_event = df_events_sorted["GlobalEventID"]
        df_events_sorted = df_events_sorted.drop("GlobalEventID",axis = 1)

        if self.label == "source": 
            df_mentions['MentionIdentifier'] = df_mentions['MentionIdentifier'].map(label_mapping_article)
            df_mentions['MentionSourceName'] = df_mentions['MentionSourceName'].map(label_mapping_source)
        elif self.label == "article":
            df_mentions['MentionSourceName'] = df_mentions['MentionSourceName'].map(label_mapping_article)
            df_mentions['MentionIdentifier'] = df_mentions['MentionIdentifier'].map(label_mapping_source)
            
        df_mentions['GlobalEventID'] = df_mentions['GlobalEventID'].map(label_mapping_event)
        df_mentions = df_mentions.dropna(subset = ["GlobalEventID"]) # because we encoded on events -> try to change that
        est_source_de = df_mentions[["MentionSourceName","MentionIdentifier"]]

        if self.label == "source":
            article_map = mapping_article.reset_index().set_index("MentionIdentifier").to_dict()
            est_source_de["MentionIdentifier"] = est_source_de["MentionIdentifier"].map(article_map["index"]).astype(int)
            source_map = mapping_source.reset_index().set_index("links").to_dict()
            est_source_de["MentionSourceName"] = est_source_de["MentionSourceName"].map(source_map["index"]).astype(int)
        
        elif self.label == "article":
            article_map = mapping_article.reset_index().set_index("MentionSourceName").to_dict()
            est_source_de["MentionSourceName"] = est_source_de["MentionSourceName"].map(article_map["index"]).astype(int)
            source_map = mapping_source.reset_index().set_index("links").to_dict()
            est_source_de["MentionIdentifier"] = est_source_de["MentionIdentifier"].map(source_map["index"]).astype(int)

        edge_est_source_de = est_source_de[["MentionSourceName", "MentionIdentifier"]].values.transpose()

        # Mapping articles and events to create edges between these two

        mentionné = df_mentions[["GlobalEventID","MentionIdentifier"]]

        if self.label == "source":
            article_map = mapping_article.reset_index().set_index("MentionIdentifier").to_dict()
            mentionné["MentionIdentifier"] = mentionné["MentionIdentifier"].map(article_map["index"]).astype(int)
            event_map = mapping_source.reset_index().set_index("links").to_dict()

        elif self.label == "article":
            source_map = mapping_source.reset_index().set_index("links").to_dict()
            mentionné["MentionIdentifier"] = mentionné["MentionIdentifier"].map(source_map["index"]).astype(int)
            event_map = mapping_source.reset_index().set_index("links").to_dict()

        mentionné["GlobalEventID"] = mentionné["GlobalEventID"].map(event_map["index"]).astype(int)
        edge_mentionné = mentionné[["GlobalEventID", "MentionIdentifier"]].values.transpose()

        # Create attributes for the mentionné edges

        df_mentions_edges = df_mentions.drop(["GlobalEventID","MentionIdentifier","MentionSourceName"], axis = 1)

        # temporarily remove almost all columns for simplicity
        df_events_sorted_temp = df_events_sorted[["Day"]] 
        df_mentions_edges_temp = df_mentions_edges[["EventTimeDate"]]
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
                return df_article_sorted,labels_sorted,df_events_sorted_temp,edge_mentionné,edge_est_source_de,y
            if self.label == "article":
                return labels_sorted,df_article_sorted,df_events_sorted_temp,edge_mentionné,edge_est_source_de,y
        else:
            return data_undirected
# def create_graph(labels,df_mentions,df_events):
    
#     # Encoding and creating the map for df_sources

#     label_encoder_source = LabelEncoder()
#     labels['links'] = label_encoder_source.fit_transform(labels['links'])
#     label_mapping_source = dict(
#         zip(label_encoder_source.classes_, label_encoder_source.transform(label_encoder_source.classes_)))
#     # labels['links'] = labels['links'].map(label_mapping_source)

#     labels_sorted = labels.sort_values(by="links").set_index("links")
#     labels_sorted = labels_sorted.reset_index(drop=False)
#     mapping_source = labels_sorted["links"]
#     labels_sorted["random"] = np.random.rand(14442)
#     y = labels_sorted["is_fake"]
#     labels_sorted = labels_sorted[["random"]]

#     df_article = pd.DataFrame(df_mentions["MentionIdentifier"])

#     label_encoder_article = LabelEncoder()
#     df_article['MentionIdentifier'] = label_encoder_article.fit_transform(df_article['MentionIdentifier'])
#     label_mapping_article = dict(
#         zip(label_encoder_article.classes_, label_encoder_article.transform(label_encoder_article.classes_)))

#     df_article_sorted = df_article.sort_values(by="MentionIdentifier").set_index("MentionIdentifier")
#     df_article_sorted = df_article_sorted.reset_index(drop=False)
#     mapping_article = df_article_sorted["MentionIdentifier"]
#     df_article_sorted["random"] = np.random.rand(1855)
#     df_article_sorted = df_article_sorted[["random"]]

#     label_encoder_event = LabelEncoder()
#     df_events['GlobalEventID'] = label_encoder_event.fit_transform(df_events['GlobalEventID'])
#     label_mapping_event = dict(
#         zip(label_encoder_event.classes_, label_encoder_event.transform(label_encoder_event.classes_)))
#     #df_events['GlobalEventID'] = df_events['GlobalEventID'].map(label_mapping_event)
#     df_events_sorted = df_events.sort_values(by="GlobalEventID").set_index("GlobalEventID")
#     df_events_sorted = df_events_sorted.reset_index(drop=False)
#     mapping_event = df_events_sorted["GlobalEventID"]
#     df_events_sorted = df_events_sorted.drop("GlobalEventID", axis=1)

#     label_encoder_event = LabelEncoder()
#     df_events['GlobalEventID'] = label_encoder_event.fit_transform(df_events['GlobalEventID'])
#     label_mapping_event = dict(
#         zip(label_encoder_event.classes_, label_encoder_event.transform(label_encoder_event.classes_)))
#     # df_events['GlobalEventID'] = df_events['GlobalEventID'].map(label_mapping_event)
#     df_events_sorted = df_events.sort_values(by="GlobalEventID").set_index("GlobalEventID")
#     df_events_sorted = df_events_sorted.reset_index(drop=False)
#     mapping_event = df_events_sorted["GlobalEventID"]
#     df_events_sorted = df_events_sorted.drop("GlobalEventID", axis=1)

#     # Mapping articles and sources to create edges between these two

#     est_source_de = df_mentions[["MentionSourceName", "MentionIdentifier"]]

#     article_map = mapping_article.reset_index().set_index("MentionIdentifier").to_dict()
#     est_source_de["MentionIdentifier"] = est_source_de["MentionIdentifier"].map(article_map["index"]).astype(int)

#     source_map = mapping_source.reset_index().set_index("links").to_dict()
#     est_source_de["MentionSourceName"] = est_source_de["MentionSourceName"].map(source_map["index"]).astype(int)

#     edge_est_source_de = est_source_de[["MentionSourceName", "MentionIdentifier"]].values.transpose()

#     # Mapping articles and events to create edges between these two

#     mentionné = df_mentions[["GlobalEventID", "MentionIdentifier"]]

#     article_map = mapping_article.reset_index().set_index("MentionIdentifier").to_dict()
#     mentionné["MentionIdentifier"] = mentionné["MentionIdentifier"].map(article_map["index"]).astype(int)

#     event_map = mapping_source.reset_index().set_index("links").to_dict()
#     mentionné["GlobalEventID"] = mentionné["GlobalEventID"].map(event_map["index"]).astype(int)

#     edge_mentionné = mentionné[["GlobalEventID", "MentionIdentifier"]].values.transpose()

#     # create attributes for the mentionné edges

#     df_mentions_edges = df_mentions.drop(["GlobalEventID", "MentionIdentifier", "MentionSourceName"], axis=1)
#     # Using only the first csv of events mentions and sources to start

#     # temporarily remove almost all columns for simplicity
#     df_events_sorted_temp = df_events_sorted[["Day"]]
#     df_mentions_edges_temp = df_mentions_edges[["EventTimeDate"]]

#     data = HeteroData()
#     data['article'].x = torch.from_numpy(df_article_sorted.to_numpy())
#     data['source'].x = torch.from_numpy(labels_sorted.to_numpy())
#     data['event'].x = torch.from_numpy(df_events_sorted_temp.to_numpy())

#     data['event', 'mentionne', 'article'].edge_attr = torch.from_numpy(df_mentions_edges_temp.to_numpy())

#     data['event', 'mentionne', 'article'].edge_index = torch.from_numpy(edge_mentionné)
#     data['source', 'est_source_de', 'article'].edge_index = torch.from_numpy(edge_est_source_de)

#     return data