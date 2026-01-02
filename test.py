#A simple test file that uploads TrainAnnouncement data from a csv file to hopsworks, and reads back what was stored there.
#The csv file is generated using the notebook

import datetime
import pandas as pd
import requests
import hopsworks
import os
import hsfs
from typing import List, Dict, Any
import hopsworks_utils
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv("out.csv")
print(df.head())



target_values: pd.DataFrame = df[["event_time", "AdvertisedTrainIdent", "Canceled", "LocationSignature", "hour_of_day", "day_of_week"]]
target_values.rename(columns={"event_time": "time"}, inplace=True)
print(target_values.info())


#Change type of the loaded dataframe
target_values["time"] = pd.to_datetime(target_values["time"])
target_values["AdvertisedTrainIdent"] = target_values["AdvertisedTrainIdent"].astype("str")
target_values["hour_of_day"] = pd.to_numeric(target_values["hour_of_day"], downcast="integer")
target_values["day_of_week"] = pd.to_numeric(target_values["day_of_week"], downcast="integer")


client = hopsworks_utils.HopsworksInterface()

client.push(target_values, "test_features", ['AdvertisedTrainIdent', 'time'])

fg_data = client.get("test_features")
print(fg_data.info())