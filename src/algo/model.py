import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression


def baseline(df):
    return df['close'].mean()
