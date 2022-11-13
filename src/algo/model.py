import logging

import pandas, numpy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import balanced_accuracy_score

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model

import pickle
import os.path


def baseline(df):
    return df['close'].mean()


def arima_model(train_df, test_cnt):
    model = ARIMA(train_df.close.values, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit.forecast(steps=test_cnt)


def model_balance_accuracy(y, y_hat):
    # print(f'y: {y} ')
    # print(f'y_hat: {y_hat}')
    return balanced_accuracy_score(y, y_hat)


def lstm_model(ticker, trainX, trainY, look_back):

    if os.path.exists(f'model_files/{ticker}.h5'):
        print(f'pre-trained model exists for {ticker}... using that for prediction')
        model = load_model(f'model_files/{ticker}.h5')
        # model = pickle.load(open(f'model_files/{ticker}.pkl', 'rb'))
    else:
        print(f'Training a new model for {ticker}...')
        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=75, batch_size=1, verbose=2)

        model.save(f'model_files/{ticker}.h5')
        # pickle.dump(model, open(f'model_files/{ticker}.pkl', 'wb'))

    return model


def lstm_pred(model, data):

    data = numpy.reshape(data, (data.shape[0], 1, data.shape[1]))
    # print(f'data is {data} and shape is {data.shape}')

    return model.predict(data)


def scale(ticker, df, scl=None):
    if scl is None:
        if os.path.exists(f'model_files/{ticker}_scalar.pkl'):
            print(f'Using the existing scalar file for {ticker}')
            scl = pickle.load(open(f'model_files/{ticker}_scalar.pkl', 'rb'))
        else:
            scl = MinMaxScaler(feature_range=(0, 1))
            print(f'here is the dataframe {df}')
            scl = scl.fit(df)
            print(f'Created a new scalar file for {ticker}')
            pickle.dump(scl, open(f'model_files/{ticker}_scalar.pkl', 'wb'))

    # print(f'columns = {df.columns} and index = {df.index.get_level_values(0)}')
    df_scaled = pandas.DataFrame(scl.transform(df), columns=df.columns, index=df.index)
    return df_scaled

