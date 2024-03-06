import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch_geometric.data import HeteroData
import torch

def create_graph(df_sources_mixte,df_test,df_test_event):
    # Encoding and creating the map for df_sources

    label_encoder_source = LabelEncoder()
    df_sources_mixte['links'] = label_encoder_source.fit_transform(df_sources_mixte['links'])
    label_mapping_source = dict(
        zip(label_encoder_source.classes_, label_encoder_source.transform(label_encoder_source.classes_)))
    # df_sources_mixte['links'] = df_sources_mixte['links'].map(label_mapping_source)

    df_sources_mixte_sorted = df_sources_mixte.sort_values(by="links").set_index("links")
    df_sources_mixte_sorted = df_sources_mixte_sorted.reset_index(drop=False)
    mapping_source = df_sources_mixte_sorted["links"]
    df_sources_mixte_sorted["random"] = np.random.rand(14442)
    y = df_sources_mixte_sorted["is_fake"]
    df_sources_mixte_sorted = df_sources_mixte_sorted[["random"]]

    df_article = pd.DataFrame(df_test["MentionIdentifier"])

    label_encoder_article = LabelEncoder()
    df_article['MentionIdentifier'] = label_encoder_article.fit_transform(df_article['MentionIdentifier'])
    label_mapping_article = dict(
        zip(label_encoder_article.classes_, label_encoder_article.transform(label_encoder_article.classes_)))

    df_article_sorted = df_article.sort_values(by="MentionIdentifier").set_index("MentionIdentifier")
    df_article_sorted = df_article_sorted.reset_index(drop=False)
    mapping_article = df_article_sorted["MentionIdentifier"]
    df_article_sorted["random"] = np.random.rand(1855)
    df_article_sorted = df_article_sorted[["random"]]

    label_encoder_event = LabelEncoder()
    df_test_event['GlobalEventID'] = label_encoder_event.fit_transform(df_test_event['GlobalEventID'])
    label_mapping_event = dict(
        zip(label_encoder_event.classes_, label_encoder_event.transform(label_encoder_event.classes_)))
    #df_test_event['GlobalEventID'] = df_test_event['GlobalEventID'].map(label_mapping_event)
    df_test_event_sorted = df_test_event.sort_values(by="GlobalEventID").set_index("GlobalEventID")
    df_test_event_sorted = df_test_event_sorted.reset_index(drop=False)
    mapping_event = df_test_event_sorted["GlobalEventID"]
    df_test_event_sorted = df_test_event_sorted.drop("GlobalEventID", axis=1)

    label_encoder_event = LabelEncoder()
    df_test_event['GlobalEventID'] = label_encoder_event.fit_transform(df_test_event['GlobalEventID'])
    label_mapping_event = dict(
        zip(label_encoder_event.classes_, label_encoder_event.transform(label_encoder_event.classes_)))
    # df_test_event['GlobalEventID'] = df_test_event['GlobalEventID'].map(label_mapping_event)
    df_test_event_sorted = df_test_event.sort_values(by="GlobalEventID").set_index("GlobalEventID")
    df_test_event_sorted = df_test_event_sorted.reset_index(drop=False)
    mapping_event = df_test_event_sorted["GlobalEventID"]
    df_test_event_sorted = df_test_event_sorted.drop("GlobalEventID", axis=1)

    # Mapping articles and sources to create edges between these two

    est_source_de = df_test[["MentionSourceName", "MentionIdentifier"]]

    article_map = mapping_article.reset_index().set_index("MentionIdentifier").to_dict()
    est_source_de["MentionIdentifier"] = est_source_de["MentionIdentifier"].map(article_map["index"]).astype(int)

    source_map = mapping_source.reset_index().set_index("links").to_dict()
    est_source_de["MentionSourceName"] = est_source_de["MentionSourceName"].map(source_map["index"]).astype(int)

    edge_est_source_de = est_source_de[["MentionSourceName", "MentionIdentifier"]].values.transpose()

    # Mapping articles and events to create edges between these two

    mentionné = df_test[["GlobalEventID", "MentionIdentifier"]]

    article_map = mapping_article.reset_index().set_index("MentionIdentifier").to_dict()
    mentionné["MentionIdentifier"] = mentionné["MentionIdentifier"].map(article_map["index"]).astype(int)

    event_map = mapping_source.reset_index().set_index("links").to_dict()
    mentionné["GlobalEventID"] = mentionné["GlobalEventID"].map(event_map["index"]).astype(int)

    edge_mentionné = mentionné[["GlobalEventID", "MentionIdentifier"]].values.transpose()

    # create attributes for the mentionné edges

    df_mentions_edges = df_test.drop(["GlobalEventID", "MentionIdentifier", "MentionSourceName"], axis=1)
    # Using only the first csv of events mentions and sources to start

    # temporarily remove almost all columns for simplicity
    df_test_event_sorted_temp = df_test_event_sorted[["Day"]]
    df_mentions_edges_temp = df_mentions_edges[["EventTimeDate"]]

    data = HeteroData()
    data['article'].x = torch.from_numpy(df_article_sorted.to_numpy())
    data['source'].x = torch.from_numpy(df_sources_mixte_sorted.to_numpy())
    data['event'].x = torch.from_numpy(df_test_event_sorted_temp.to_numpy())

    data['event', 'mentionne', 'article'].edge_attr = torch.from_numpy(df_mentions_edges_temp.to_numpy())

    data['event', 'mentionne', 'article'].edge_index = torch.from_numpy(edge_mentionné)
    data['source', 'est_source_de', 'article'].edge_index = torch.from_numpy(edge_est_source_de)

    return data