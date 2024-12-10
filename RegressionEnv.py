from BaseEnv import BaseClass
import tensorflow as tf
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from keras.losses import Huber, MeanSquaredError, MeanAbsoluteError, MeanSquaredLogarithmicError, CosineSimilarity
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import keras.backend as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RegressionClass(BaseClass):

    def __init__(self,
        feature_steps: int = 10,
        target_steps: int = 1,
        tickers: List[str] = ['aapl']
    ):  
        super().__init__(feature_steps = feature_steps, target_steps = target_steps, tickers=tickers)
        self.models = {}
        self.models_name_str = {LinearRegression: "LinearRegression", ARIMA: "ARIMA"}
        self.train_series = {}
        self.train_pred = {}
        self.valid_pred = {}
        self.test_pred = {}
        self.train_errors = {}
        self.valid_errors = {}
        self.test_errors = {}
        self.test_dates = {}
        self.history = {}
        for t in self.tickers:
            self.train_series[t] = np.concatenate( (self.y_train[t],self.y_valid[t],self.y_test[t]), axis=0)
            self.models[t] = dict()
            self.train_pred[t] = {LinearRegression: [], ARIMA: []}
            self.valid_pred[t] = {LinearRegression: [], ARIMA: []}
            self.test_pred[t] = {LinearRegression: [], ARIMA: []}
            self.train_errors[t] = {LinearRegression: [], ARIMA: []}
            self.valid_errors[t] = {LinearRegression: [], ARIMA: []}
            self.test_errors[t] = {LinearRegression: [], ARIMA: []}
            self.test_dates[t] = {LinearRegression: [], ARIMA: []}
            self.history[t] = {}

    def Prediction(self,
        model,
    ):
        for t in self.tickers:
            n_train = len(self.X_train[t])
            n_valid = len(self.X_valid[t])
            n_test = len(self.X_test[t])

            if model == LinearRegression:
                m = LinearRegression()
                m.fit(self.X_train[t], self.y_train[t].ravel())
                self.train_pred[t][model] = m.predict(self.X_train[t])
                self.valid_pred[t][model] = m.predict(self.X_valid[t])
                self.test_pred[t][model] = m.predict(self.X_test[t])
            else:
                raise TypeError("model must be LinearRegression or ARIMA")            

            self.train_errors[t][model] = mean_squared_error(self.y_train[t], self.train_pred[t][model])
            self.valid_errors[t][model] = mean_squared_error(self.y_valid[t], self.valid_pred[t][model])
            self.test_errors[t][model] = mean_squared_error(self.y_test[t], self.test_pred[t][model])

            self.test_dates[t][model] = self.dates[t][-n_test:]

            self.models[t][model] = m

    def VisualizationRNN(self,
        model,
        plot: bool = False,
        logdiff: bool = True
    ):
        if model not in [LinearRegression]:
            raise TypeError("model must be LinearRegression")
        else:
            self.Visualization(model=model,plot=plot,logdiff=logdiff)