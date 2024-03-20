import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
from abc import ABC, abstractmethod

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


class Preprocessing(ABC): # NOTE variable names are misleading if self.label = article

    
    def __init__(self,label):
        self.label = label
        if  self.label == "source":
            self.label_column = "MentionSourceName"
            self.non_label_column = "MentionIdentifier"
        elif self.label == 'article':
            self.label_column = "MentionIdentifier"
            self.non_label_column = "MentionSourceName"
    
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
    
    def _create_label_node(self,labels,df_mentions): 
        
        label_encoder_source = LabelEncoder()
        labels['links'] = label_encoder_source.fit_transform(labels['links'])
        label_mapping_source = dict(zip(label_encoder_source.classes_, label_encoder_source.transform(label_encoder_source.classes_)))
        labels_sorted = labels.sort_values(by="links").set_index("links")
        labels_sorted = labels_sorted.reset_index(drop=False)
        mapping_source = labels_sorted["links"]
        labels_sorted["random"] = np.random.rand(len(labels_sorted))
        y = labels_sorted["is_fake"]#.apply(lambda x:int(x) if not pd.isna(x) else x )
        labels_sorted = labels_sorted[["random"]]
        df_mentions[self.label_column] = df_mentions[self.label_column].map(label_mapping_source)
            
        return df_mentions,labels_sorted,mapping_source,y
    
    def _create_non_label_node(self,df_mentions):
        
        df_article = pd.DataFrame(df_mentions[self.non_label_column])
        label_encoder_article = LabelEncoder()
        df_article[self.non_label_column] = label_encoder_article.fit_transform(df_article[self.non_label_column])
        label_mapping_article = dict(zip(label_encoder_article.classes_, label_encoder_article.transform(label_encoder_article.classes_)))
        df_article_sorted = df_article.sort_values(by=self.non_label_column).set_index(self.non_label_column)
        df_article_sorted = df_article_sorted.reset_index(drop=False)
        mapping_article = df_article_sorted[self.non_label_column]
        df_article_sorted["random"] = np.random.rand(len(df_article_sorted))
        df_article_sorted = df_article_sorted[["random"]]
        df_mentions[self.non_label_column] = df_mentions[self.non_label_column].map(label_mapping_article)

        return df_mentions,mapping_article,df_article_sorted

    def _create_event_node(self,df_events,df_mentions):
        
        label_encoder_event = LabelEncoder()
        df_events['GlobalEventID'] = label_encoder_event.fit_transform(df_events['GlobalEventID'])
        label_mapping_event = dict(zip(label_encoder_event.classes_, label_encoder_event.transform(label_encoder_event.classes_)))
        # df_events['GlobalEventID'] = df_events['GlobalEventID'].map(label_mapping_event)
        # The three next lines seems to not be important
        df_events_sorted = df_events.sort_values(by="GlobalEventID").set_index("GlobalEventID")
        df_events_sorted = df_events_sorted.reset_index(drop=False)
        df_events_sorted = df_events_sorted.drop("GlobalEventID",axis = 1)
        df_mentions['GlobalEventID'] = df_mentions['GlobalEventID'].map(label_mapping_event)
        
        return df_mentions,df_events, df_events_sorted
  
    def _create_est_source_de_edge(self,df_mentions,mapping_article,mapping_source):
        
        df_mentions = df_mentions.dropna(subset = ["GlobalEventID"]) # because we encoded on events -> try to change that
        est_source_de = df_mentions[["MentionSourceName","MentionIdentifier"]]

        article_map = mapping_article.reset_index().set_index(self.non_label_column).to_dict()
        est_source_de[self.non_label_column] = est_source_de[self.non_label_column].map(article_map["index"]).astype(int)
        source_map = mapping_source.reset_index().set_index("links").to_dict()
        est_source_de[self.label_column] = est_source_de[self.label_column].map(source_map["index"]) # Bugfix attempt, may be suboptimal
        est_source_de = est_source_de.dropna(subset=[self.label_column])
        est_source_de[self.label_column] = est_source_de[self.label_column].astype(int)
        edge_est_source_de = est_source_de[["MentionSourceName", "MentionIdentifier"]].values.transpose()
    
        return edge_est_source_de,df_mentions
    
    def _create_mentionne_edge(self,df_mentions,mapping_article,mapping_source):
        
        mentionné = df_mentions[["GlobalEventID","MentionIdentifier"]]

        if self.label == "source":
            article_map = mapping_article.reset_index().set_index("MentionIdentifier").to_dict()
            mentionné["MentionIdentifier"] = mentionné["MentionIdentifier"].map(article_map["index"]).astype(int)

        elif self.label == "article":
            source_map = mapping_source.reset_index().set_index("links").to_dict()
            mentionné["MentionIdentifier"] = mentionné["MentionIdentifier"].map(source_map["index"]).astype(int)

        event_map = mapping_source.reset_index().set_index("links").to_dict()
        mentionné["GlobalEventID"] = mentionné["GlobalEventID"].map(event_map["index"])
        mentionné = mentionné.dropna(subset=["GlobalEventID"]) # Bugfix attempt, may be suboptimal
        mentionné["GlobalEventID"] = mentionné["GlobalEventID"].astype(int)
        # print(mentionné["GlobalEventID"].isna().value_counts())
        edge_mentionné = mentionné[["GlobalEventID", "MentionIdentifier"]].values.transpose()
    
        return edge_mentionné,event_map
    
    @abstractmethod
    def create_graph(self):
        pass
    


        
