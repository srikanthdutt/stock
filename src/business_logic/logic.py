import configparser
import logging

import joblib
import pandas

from src.IO.import_data import *
from src.IO.save_model import get_model_from_gcp_bucket, upload_file_to_bucket
from src.algo.model import *
from datetime import datetime, timedelta, date
from statistics import mean

logging.basicConfig(filename='Log.txt', filemode='w', format='%(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


def get_prediction(ticker, mname='baseline', days_cnt=30, return_accuracy=False, look_back=15):
    data = get_info(ticker, days_cnt)
    pred = ''

    test_st_dt = '2022-06-01'
    test_end_dt = '2022-09-01'

    if isinstance(data, pandas.DataFrame):
        data = data['close'].to_frame()
        bucket_nm = ''

        pred_val = []
        pred_for = date.today() + timedelta(days=1)
        train, test = split_train_test(data, test_st_dt, test_end_dt)

        if mname == 'baseline':
            pred = baseline(data)
            logging.debug(f'predicted value is {pred}')
        elif mname == 'arima':

            pred_val = arima_model(train, abs((datetime.strptime(test_st_dt, '%Y-%m-%d').date() - pred_for).days))
            pred = pred_val[-1]

        elif mname == 'lstm':
            bucket_nm = 'sri_new_bucket_lstm'
            # print(f'############## {data.index} before')
            data = shift_close(data.copy(), look_back)
            # print(f'############## {data.index} after')

            logging.info(f'Checking if the MinMax Scalar model already exists for {ticker}_scalar in GCP')
            scl_model = get_model_from_gcp_bucket(f'{ticker}_scalar.pkl', bucket_nm)
            no_scalar_on_gcp = 0 if scl_model is None else 0

            data = scale(ticker, data, scl_model)
            data.reset_index(inplace=True)
            data.set_index('index', inplace=True)

            if no_scalar_on_gcp:
                logging.info(f'saving MinMax Scalar model for {ticker}_scalar on gcp in {bucket_nm}')
                # The file is being saved in the below location by model.py
                upload_file_to_bucket(f'model_files/{ticker}._scalar.pkl', bucket_nm)

            train, test = split_train_test(data, test_st_dt, test_end_dt)
            # print(f'train data is {train} and columns are {train.columns}')
            # print(f"trainX should be {train.drop(columns='close')}")

            trainX, trainY = train.drop(columns='close').values,  train['close'].values
            testX, testY = test.drop(columns='close').values, test['close'].values

            logging.info(f'Checking if the model already exists for {ticker} in GCP')
            model = get_model_from_gcp_bucket(f'{ticker}.h5', bucket_nm)

            if model is None:
                logging.info(f'training model for {ticker}')
                model = lstm_model(ticker, trainX, trainY, look_back)
                logging.info(f'saving model for {ticker} on gcp in {bucket_nm}')

                # The file is being saved in the below location by model.py
                upload_file_to_bucket(f'model_files/{ticker}.h5', bucket_nm)

            # preparing the data for prediction of tomorrow
            # to do that.. gathering the last <look_back> number of days.. and labelling them with lag
            # iloc[-look_back:, 0] should get us just the close value(0th col) and look_back no of rows
            col_map = dict(zip(data.iloc[-look_back:, 0].to_frame().sort_index(ascending=False).index,
                               [f'lag_{item}' for item in range(1, look_back + 1)]))
            data_for_pred = data.iloc[-look_back:, 0].to_frame().sort_index(ascending=False).T.rename(columns=col_map)

            # print(f 'column map is {col_map} and data is {data_for_pred.values}')

            logging.info(f'predicting the values for tomorrow')
            pred = lstm_pred(model, data_for_pred.values)

            logging.info(f'predicting the values for test')
            pred_val = lstm_pred(model, testX)
            logging.debug(f'predicted value for test is {pred_val}')

            logging.info(f'predicting the values for train')
            pred_train_val = lstm_pred(model, trainX)
            logging.debug(f'predicted value for test is {pred_val}')

        logging.debug(f'predicted value for tomorrow is {pred}')

        # print(f'pred_val type is {pred_val} and index of test is {test.index} and '
        #       f'shape of pred_val is {pred_val.shape}............')
        test_hat = pandas.DataFrame(pred_val[0:len(test)], columns=['close'],  index=test.index)
        train_hat = pandas.DataFrame(pred_train_val, columns=['close'], index=train.index)

        b_accuracy_test = model_balance_accuracy(create_y(test).target, create_y(test_hat).target)
        b_accuracy_train = model_balance_accuracy(create_y(train).target, create_y(train_hat).target)
        reco = 'Sell' if data['close'].iloc[-1] > pred else 'Buy'

        print_val = f"<b> Ticker: {ticker}</b><br>" \
                    f"tomorrow's predicted {pred} <br> today's close {data['close'].iloc[-1]} <br><br>" \
                    f"<b> Recommendation: {reco}</b><br>" \
                    f"Train Balance Accuracy: {b_accuracy_train} and Test Balance Accuracy: {b_accuracy_test}"

        logging.info(print_val)

        # replace print_val with reco when done
        return (b_accuracy_train, b_accuracy_test) if return_accuracy is True else reco

    else:
        return data


def split_train_test(df, test_st_dt='2022-06-01', test_end_dt='2022-09-01'):
    return (df.loc[:datetime.strptime(test_st_dt, '%Y-%m-%d') - timedelta(days=1)],
            df.loc[test_st_dt: datetime.strptime(test_end_dt, '%Y-%m-%d') - timedelta(days=1)])


def create_y(df):

    df['tmrw'] = df['close'].shift(-1)
    df.dropna(inplace=True)

    # print(f"df.close is {df['close'].index} and df.tmrw is {df['tmrw'].index}")
    # Sell = 0, Buy = 1
    # if today's price (close) is less than tomorrow's value (tmrw) then Buy
    df['target'] = 0
    df.loc[df['close'] < df['tmrw'], 'target'] = 1

    return df


def get_overall_accuracy(mname='baseline', days_cnt=30):

    tickers = get_sp500()
    train_acc_list = []
    test_acc_list = []
    logging.info(f'model: {mname}\n Time period used: {days_cnt}')
    for ticker in tickers:
        train_acc, test_acc = get_prediction(ticker, mname, days_cnt, return_accuracy=True)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        logging.info(f'{ticker} has balance accuracy score of {train_acc} on train dataset '
                     f'and {test_acc} on test dataset')

    logging.info(f'accuracy by tickers: {tickers} ---------------{train_acc_list}---------------------{test_acc_list}')
    logging.info(f'the overall test accuracy is {mean(test_acc_list)}')
    return mean(test_acc_list)


def shift_close(train_df, shift_cnt):

    if shift_cnt >= 1:
        for lag in range(1, shift_cnt+1):
            train_df[f'lag_{lag}'] = train_df['close'].shift(lag)

    train_df.reset_index(inplace=True)

    return train_df.set_index('index').dropna()
