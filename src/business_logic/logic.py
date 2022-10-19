import configparser
import logging

import joblib
import pandas

from src.IO.import_data import *
# from src.IO.storage_tools import create_bucket, get_model_from_bucket, upload_file_to_bucket
from src.algo.model import *


def get_prediction(ticker):
    data = get_info(ticker, days_cnt=30)
    if isinstance(data, pandas.DataFrame):
        pred = baseline(data)
        # print(f'predicted value is {pred}')
        reco = 'Sell' if data['close'].iloc[-1] > pred else 'Buy'
        return f"tomorrow's predicted {pred} <br> today's close {data['close'].iloc[-1]} <br><br> " \
               f"<b>Recommendation: {reco}</b>"
    else:
        return data
