import configparser
import logging

import joblib
import pandas

from src.IO.import_data import *
# from src.IO.storage_tools import create_bucket, get_model_from_bucket, upload_file_to_bucket
from src.algo.model import *
from datetime import datetime, timedelta, date


def get_prediction(ticker, mname='baseline', days_cnt=30):
    data = get_info(ticker, days_cnt)
    pred = ''
    b_accuracy = 0

    test_st_dt = '2022-06-01'
    test_end_dt = '2022-09-01'

    if isinstance(data, pandas.DataFrame):

        if mname == 'baseline':
            pred = baseline(data)
            # print(f 'predicted value is {pred}')
        elif mname == 'arima':

            train, test = split_train_test(data, test_st_dt, test_end_dt)
            pred_for = date.today() + timedelta(days=1)

            pred_val = arima_model(train, abs((datetime.strptime(test_st_dt, '%Y-%m-%d').date() - pred_for).days))
            pred = pred_val[-1]

            test_hat = pandas.DataFrame(pred_val[0:len(test)], columns=['close'],  index=[test.index])

            b_accuracy = model_balance_accuracy(create_y(test).target, create_y(test_hat).target)

        reco = 'Sell' if data['close'].iloc[-1] > pred else 'Buy'

        print_val = f"tomorrow's predicted {pred} <br> today's close {data['close'].iloc[-1]} <br><br>" \
                    f"<b> Recommendation: {reco}</b><br>" \
                    f"Balance Accuracy: {b_accuracy}"

        return print_val

    else:
        return data


def split_train_test(df, test_st_dt='2022-06-01', test_end_dt='2022-09-01'):
    return (df.loc[:datetime.strptime(test_st_dt, '%Y-%m-%d') - timedelta(days=1)],
            df.loc[test_st_dt: datetime.strptime(test_end_dt, '%Y-%m-%d') - timedelta(days=1)])


def create_y(df):
    df['tmrw'] = df['close'].shift(1)
    df.dropna(inplace=True)

    # Sell = 0, Buy = 1
    # if today's price (close) is less than tomorrow's value (tmrw) then Buy
    df['target'] = 0
    df.loc[df['close'] < df['tmrw'], 'target'] = 1

    return df
