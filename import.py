import json
import os
 
from newsplease import NewsPlease
import pandas as pd
import pickle
import requests
import tldextract
from tqdm import tqdm
import zipfile
import numpy as np
 
 # This file is used to get data from GDELT and label it
 
 
if __name__=="__main__":
 
    years = ["2023"]#
    months = ["10"]#,"02","03","04","05","06","07","08","09","10","11","12"]
    days = [["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"]
            ]
    hours = ["00","01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23"]
    minutes = ["0000","1500","3000","4500"]
    
    for year in years:
        for i in range(len(months)):
            month = months[i]
            days_of_month = days[i]
            for day in tqdm(days_of_month):
                for hour in hours:
                    for minute in minutes:
                        try: 
                            url = "http://data.gdeltproject.org/gdeltv2/"+year+month+day+hour+minute+".export.CSV.zip"
                            r = requests.get(url, allow_redirects=True)
                            open('gdelt_data/'+year+month+day+hour+minute+".export.CSV.zip", 'wb').write(r.content)
                            with zipfile.ZipFile('gdelt_data/'+year+month+day+hour+minute+".export.CSV.zip", 'r') as zip_ref:
                                    zip_ref.extractall("gdelt_data_event/")
                            os.remove('gdelt_data/'+year+month+day+hour+minute+".export.CSV.zip")
                        except:
                            print(f"Failed for this one ({year+month+day+hour+minute})")
 
    data = pd.read_csv("../mediabiasfactcheck/mediabiasfactcheck.csv")
    unreliable_labels = ['MBFCLow.png', 'MBFCVeryLow.png']
    reliable_labels = ["MBFCMixed.png","MBFCHigh.png","MBFCMostlyFactual.png","MBFCVeryhigh.png"]
    unreliable_ = data[data['image_factual'] == 'MBFCLow.png']["url"].unique().tolist()
    unreliable_ = unreliable_ + data[data['image_factual'] == 'MBFCVeryLow.png']["url"].unique().tolist()
    # unreliable_ = unreliable_ + data[data['image_factual'] == "MBFCMixed.png"]["url"].unique().tolist()
    reliable_ = data[data['image_factual'].isin(reliable_labels)]["url"].unique().tolist()
    unreliable = []
    reliable = []
    for x in unreliable_:
        if type(x) == str:
            unreliable.append(x)
    for x in reliable_:
        if type(x) == str:
            reliable.append(x)
    unreliable_clean = []
    reliable_clean = []
    for i in tqdm(range(len(unreliable))):
        unreliable_clean.append(tldextract.extract(unreliable[i]).registered_domain)
    for i in tqdm(range(len(reliable))):
        reliable_clean.append(tldextract.extract(reliable[i]).registered_domain)
       
    col_names_events = ["GlobalEventID","Day","MonthYear","Year","FractionDate",
                            "Actor1Code","Actor1Name","Actor1CountryCode","Actor1KnownGroupCode","Actor1EthnicCode",
                            "Actor1Religion1Code","Actor1Religion2Code","Actor1Type1Code","Actor1Type2Code","Actor1Type3Code",
                            "Actor2Code","Actor2Name","Actor2CountryCode","Actor2KnownGroupCode","Actor2EthnicCode",
                            "Actor2Religion1Code","Actor2Religion2Code","Actor2Type1Code","Actor2Type2Code","Actor2Type3Code",
                            "IsRootEvent","EventCode","EventBaseCode","EventRootCode","QuadClass",
                            "GoldsteinScale","NumMentions","NumSources","NumArticles","AvgTone",
                            "Actor1Geo_Type","Actor1Geo_Fullname","Actor1Geo_CountryCode","Actor1Geo_ADM1Code","Actor1Geo_ADM2Code",
                            "Actor1Geo_Lat","Actor1Geo_Long","Actor1Geo_FeatureID","Actor2Geo_Type","Actor2Geo_Fullname",
                            "Actor2Geo_CountryCode","Actor2Geo_ADM1Code","Actor2Geo_ADM2Code","Actor2Geo_Lat","Actor2Geo_Long",
                            "Actor2Geo_FeatureID","ActionGeo_Type","ActionGeo_Fullname","ActionGeo_CountryCode","ActionGeo_ADM1Code",
                            "ActionGeo_ADM2Code","ActionGeo_Lat","ActionGeo_Long","ActionGeo_FeatureID","DATEADDED",
                            "SOURCEURL"]
 
    col_names_mentions = ["GlobalEventID","EventTimeDate","MentionTimeDate","MentionType",
                "MentionSourceName","MentionIdentifier","SentenceID","Actor1CharOffset","Actor2CharOffset",
                "ActionCharOffset","InRawText","Confidence","MentionDocLen","MentionDocTone",
                "SRCLC", "ENG"]
 
    urls_total = []
    is_fake_total = []
    for year in years:
        for i in range(len(months)):
            month = months[i]
            days_of_month = days[i]
            for day in tqdm(days_of_month):
                for hour in hours:
                    for minute in minutes:
                        try:
                            gdelt_events = pd.read_csv("gdelt_data/"+year+month+day+hour+minute+".mentions.CSV", delimiter='\t', names=col_names_mentions)
                            urls = gdelt_events["MentionIdentifier"].tolist()# MentionSourceName MentionIdentifier
                            urls_clean = [tldextract.extract(url).registered_domain for url in urls]
                            urls_to_process = []
                            is_fake = []
                            for i in range(len(urls_clean)):
                                url = urls_clean[i]
                                if url in unreliable_clean:
                                    is_fake.append(1)
                                elif url in reliable_clean:
                                    is_fake.append(0)
                                else:
                                    is_fake.append(np.nan)
                                urls_to_process.append(urls[i])
                            urls_total = urls_total + urls_to_process
                            is_fake_total = is_fake_total + is_fake
                            # for i, url in enumerate(urls_to_process):
                            #     try:
                            #         article = NewsPlease.from_url(url)
                            #         with open("articles/"+year+month+day+hour+minute+"_"+str(i)+".json", "w") as file:
                            #             json.dump(article.get_serializable_dict(), file)
                            #     except:
                            #         continue
                        except:
                            continue
    print(len(is_fake_total),len(urls_total))
    df = pd.DataFrame({"links":urls_total,"is_fake":is_fake_total})
    df = df.drop_duplicates().reset_index(drop=True)                  
    pickle.dump(df, open("large_articles_labels.pkl", "wb"))
                       
    print(f"{len(df)} articles found")
    
    