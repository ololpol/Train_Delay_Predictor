import datetime
import pandas as pd
import requests
import hopsworks
import os
import hsfs
from dotenv import load_dotenv
from typing import List, Dict, Any

class HopsworksInterface():


    def __init__(self):
        load_dotenv()

        # 1) Where your Hopsworks UI lives (domain in your browser)
        os.environ["HOPSWORKS_HOST"] = "eu-west.cloud.hopsworks.ai"
        os.environ["HOPSWORKS_PORT"] = "443"

        # 2) Hopsworks login key (must be a valid Hopsworks API key)

        HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
        self.project_name = os.getenv("HOPSWORKS_PROJECT")

        print("HOPSWORKS_API_KEY exists:", HOPSWORKS_API_KEY is not None)
        print("HOPSWORKS_API_KEY length:", len(HOPSWORKS_API_KEY.strip()) if HOPSWORKS_API_KEY else None)

        assert len(os.environ["HOPSWORKS_API_KEY"]) > 20, "Missing/invalid HOPSWORKS_API_KEY in Colab Secrets"
        assert HOPSWORKS_API_KEY, "Missing HOPSWORKS_API_KEY secret"
        self.project = hopsworks.login()
        self.fs = self.project.get_feature_store(self.project_name)

    def push(self, df: pd.DataFrame, fg_name: str, primary_key: str = None, desc: str = "No description provided"):
        pk = primary_key
        if primary_key == None:
            pk = [df.columns[0]]
        if df.empty:
            print('No data to insert into feature store.')
            delay_fg = self.fs.get_or_create_feature_group(
                name=fg_name,
                version=1, #TODO idk
                description=desc,
                primary_key=pk,
            )
        else:
            delay_fg = self.fs.get_or_create_feature_group(
                name=fg_name,
                version=1, #TODO idk
                description=desc,
                primary_key=pk,
            )

            #print(pk in df.columns)
            delay_fg.insert(df, write_options={'wait_for_job': True}, storage="spark")
            print('Inserted historical data into feature group "train_delay_features"')
    def get(self, fg_name) -> pd.DataFrame:
        fg = self.fs.get_feature_group(
            name=fg_name,
            version=1
        )
        df = fg.read()
        return df