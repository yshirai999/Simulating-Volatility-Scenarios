from BaseEnv import BaseClass
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout, TimeDistributed, BatchNormalization
from tensorflow.keras.models import Sequential
from keras.losses import Huber, MeanSquaredError, MeanAbsoluteError, MeanSquaredLogarithmicError, CosineSimilarity
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import keras.backend as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RNNClass(BaseClass):

    def __init__(self,
        feature_steps: int = 10,
        target_steps: int = 1,
        batchnormalization: bool = False,
        epochs: int = 500,
        patience: int = 200,
        HuberDelta: float = 0.0001,
        layers_RNN: int = 2,
        layers_LSTM: int = 2,
        tickers: List[str] = ['aapl']
    ):  
        super().__init__(feature_steps = feature_steps, target_steps = target_steps, tickers=tickers)
        self.epochs = epochs
        self.patience = patience
        self.HuberDelta = HuberDelta
        self.models = {}
        self.bn = batchnormalization
        self.layers = {SimpleRNN: layers_RNN, LSTM: layers_LSTM}
        self.models_function_name = {SimpleRNN: self.rnn_dense_model, LSTM: self.lstm_model}
        self.models_name_str = {SimpleRNN: "SimpleRNN", LSTM: "LSTM"}
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
            self.train_pred[t] = {SimpleRNN: [], LSTM: []}
            self.valid_pred[t] = {SimpleRNN: [], LSTM: []}
            self.test_pred[t] = {SimpleRNN: [], LSTM: []}
            self.train_errors[t] = {SimpleRNN: [], LSTM: []}
            self.valid_errors[t] = {SimpleRNN: [], LSTM: []}
            self.test_errors[t] = {SimpleRNN: [], LSTM: []}
            self.test_dates[t] = {SimpleRNN: [], LSTM: []}
            self.history[t] = {}

    def Prediction(self,
        model,
        verbose: bool = False
    ):
        for t in self.tickers:
            self.reset_session()
            n_train = len(self.X_train[t])
            n_valid = len(self.X_valid[t])
            n_test = len(self.X_test[t])

            input_shape = (self.X_train[t].shape[1],self.X_train[t].shape[2])
            output_units = self.y_train[t].shape[1] if len(self.y_train[t].shape) > 1 else 1

            if model in [SimpleRNN, LSTM]:
                m = self.models_function_name[model](input_shape=input_shape, output_units=output_units, layers = self.layers[model])
                m.compile(loss=Huber(delta=self.HuberDelta), optimizer="nadam")
            else:
                raise TypeError("model must be SimpleRNN or LSTM")
            

            early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=self.patience,
                                                              min_delta=0.01,
                                                              restore_best_weights=True)
            
            if verbose:
                m.summary()

            self.history[t][model] = m.fit(self.X_train[t][..., np.newaxis], self.y_train[t][..., np.newaxis], epochs=self.epochs,
                                              validation_data=(self.X_valid[t][..., np.newaxis], self.y_valid[t][..., np.newaxis]),
                                              callbacks=[early_stopping_cb], verbose=0)

            #pd.DataFrame(self.history[t][model].history).iloc[-11:]

            self.train_pred[t][model] = m.predict(self.X_train[t])
            self.valid_pred[t][model] = m.predict(self.X_valid[t])
            self.test_pred[t][model] = m.predict(self.X_test[t])

            self.train_errors[t][model] = mean_squared_error(self.y_train[t], self.train_pred[t][model])
            self.valid_errors[t][model] = mean_squared_error(self.y_valid[t], self.valid_pred[t][model])
            self.test_errors[t][model] = mean_squared_error(self.y_test[t], self.test_pred[t][model])

            self.test_dates[t][model] = self.dates[t][-n_test:]

            self.models[t][model] = m

    def rnn_dense_model(self,
        input_shape,
        output_units,
        layers
    ):
        model = Sequential()
        
        for i in range(layers):
            model.add(SimpleRNN(units=64, activation='relu', input_shape=input_shape, return_sequences=True))
            model.add(Dropout(0.2))        

            model.add(TimeDistributed(Dense(units=32, activation='relu')))

        model.add(SimpleRNN(units=64, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        
        model.add(Dense(units=64, activation='relu'))

        if self.bn:
            model.add(BatchNormalization())

        model.add(Dense(units=32, activation='relu'))
        
        model.add(Dense(units=output_units))
        return model

    def lstm_model(self,
        input_shape,
        output_units,
        layers
    ):
        model = Sequential()

        for i in range(layers):
            model.add(LSTM(units=64, input_shape=input_shape, return_sequences=True))
            model.add(Dropout(0.2))
        
        model.add(LSTM(units=64, return_sequences=False))
        
        model.add(Dense(units=output_units))
        
        return model
    
    def reset_session(self,
            seed=42
        ):
            tf.random.set_seed(seed)
            np.random.seed(seed)
            tf.keras.backend.clear_session()

    def VisualizationRNN(self,
        model,
        plot: bool = False,
        logdiff: bool = True
    ):
        if model not in [SimpleRNN,LSTM]:
            raise TypeError("model must be SimpleRNN or LSTM")
        else:
            self.Visualization(model=model,plot=plot,logdiff=logdiff)