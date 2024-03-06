import pandas as pd
import pickle

def data_load():
    col_names_mentions = ["GlobalEventID", "EventTimeDate", "MentionTimeDate", "MentionType",
                          "MentionSourceName", "MentionIdentifier", "SentenceID", "Actor1CharOffset", "Actor2CharOffset",
                          "ActionCharOffset", "InRawText", "Confidence", "MentionDocLen", "MentionDocTone",
                          "SRCLC", "ENG"]  # mention columns

    col_names_events = ["GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
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

    with open("../urls_to_download_fr.pkl", 'rb') as file:  # label of articles (0 if fake or not present in MBFS)
        df_mentions = pickle.load(file)

    with open("labeled sources.pkl", 'rb') as file:  # label of sources (0 if fake or not present in MBFS)
        df_sources = pickle.load(file)

    with open("mixt_labeled_sources.pkl", 'rb') as file:  # label of sources (nan if not present in MBFS)
        df_sources_mixte = pickle.load(file)

    with open("mixt_labeled_articles.pkl", 'rb') as file:  # label of articles (nan if not present in MBFS)
        df_articles_mixte = pickle.load(file)

    MBFS = pd.read_csv("../mediabiasfactcheck.csv")

    df_test = pd.read_csv("./20231001000000.mentions.CSV", delimiter='\t', names=col_names_mentions)

    df_test_event = pd.read_csv("./20231001000000.export.CSV", delimiter='\t', names=col_names_events)

    return df_mentions,df_sources,df_sources_mixte,df_articles_mixte,MBFS,df_test,df_test_event
