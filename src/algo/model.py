import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import balanced_accuracy_score


def baseline(df):
    return df['close'].mean()


def arima_model(train_df, test_cnt):
    model = ARIMA(train_df.close.values, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit.forecast(steps=test_cnt)


def model_balance_accuracy(y, y_hat):
    print(f'y: {y} ')
    print(f'y_hat: {y_hat}')
    return balanced_accuracy_score(y, y_hat)


