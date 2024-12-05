import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from data import dataclass
from typing import List

class BaseClass(dataclass):

    def __init__(self,
    tickers: List[str] = ['aapl'],
    feature_steps: int = 10,
    target_steps: int = 1,
    ):
        super().__init__()
        self.tickers = tickers
        self.BG = dict()
        self.dates = dict()
        for t in self.tickers:
            loc = self.tickersloc[t]
            BG = self.BGP[loc[0]]["parms"][self.BGP[loc[0]]["tickers"][loc[1]]]
            self.BG[t] = BG[:][1:-2]
            self.df[t] = pd.dataframe(self.BG[:][1:-2])
            self.dates[t] = [int(d) for d in BG[:][0]]
            self.dates[t] = [dt.strptime(d, '%Y%m%d').strftime('%m/%d/%Y') for d in self.dates]
        self.feature_steps = feature_steps
        self.target_steps = target_steps
        self.ts = {}
        self.X = {}
        self.y = {}
        self.X_train_full = {}
        self.y_train_full = {}
        self.X_test = {}
        self.y_test = {}
        self.X_train = {}
        self.y_train = {}
        self.X_valid = {}
        self.y_valid = {}
        self.split_ind = {}
        self.split_ind_2 = {}
        self.prc = {}
        self.scalers = {}
        self.test_pred_rescaled = {}
        self.y_test_rescaled = {}
        for t in self.tickers:
            data = self.df[t].diff().dropna()
            self.scalers[t] = MinMaxScaler()
            data = self.scalers[t].fit_transform(data)
            self.ts[t] = data#.values#.flatten()
            self.X[t], self.y[t] = self.ts_split(self.ts[t])
            self.split_ind[t] = int(self.X[t].shape[0]*0.8)
            self.X_train_full[t], self.y_train_full[t] = self.X[t][:self.split_ind[t]], self.y[t][:self.split_ind[t]]
            self.X_test[t], self.y_test[t] = self.X[t][self.split_ind[t]:], self.y[t][self.split_ind[t]:]
            self.split_ind_2[t] = int(self.X_train_full[t].shape[0]*0.8)
            self.X_train[t], self.y_train[t] = self.X_train_full[t][:self.split_ind_2[t]], self.y_train_full[t][:self.split_ind_2[t]]
            self.X_valid[t], self.y_valid[t] = self.X_train_full[t][self.split_ind_2[t]:], self.y_train_full[t][self.split_ind_2[t]:]
            self.y_test_rescaled[t] = self.scalers[t].inverse_transform(self.y_test[t]).flatten()
    
    def visualizedata(self
    ):
        print(self.df)
        for t in self.tickers:
            plt.plot(self.dates[t], self.df[t].values, '-', label = name)
        plt.legend()
        print(self.X_train_full['SPY'][0])
        print(self.X_train_full['SPY'][1])
        print(self.y_train_full['SPY'][0])

    def ts_split(self,
        ts
    ):
        feature_steps = self.feature_steps
        target_steps = self.target_steps
        n_obs = len(ts) - feature_steps - target_steps + 1
        X = np.array([ts[idx:idx + feature_steps].flatten() for idx in range(n_obs)])
        y = np.array([ts[idx + feature_steps:idx + feature_steps + target_steps][:, -1]
                    for idx in range(n_obs)])
        return X, y
    
    def Visualization(self,
        model,
        plot: bool = False,
        logdiff: bool = True
    ):
        try:
            if plot:
                for name in self.tickers.groups.keys():
                    plt.figure(figsize=(10, 5))
                    if logdiff:
                        plt.plot(self.test_dates[name][model], self.y_test_rescaled[name], label="Actual")
                        plt.plot(self.test_dates[name][model], self.test_pred[name][model], label="Predicted")
                    else:
                        plt.plot(self.test_dates[name][model], self.y_test_prc[name][model], label="Actual")
                        plt.plot(self.test_dates[name][model], self.y_pred_prc[name][model], label="Predicted")
                    plt.title(f"{self.models_name_str[model]}: Predicted vs Actual log difference on test dataset for {name}")
                    plt.xlabel("Time Steps")
                    plt.ylabel("Price")
                    plt.legend()
                    plt.show()

            print(f"{self.models_name_str[model]}: mean Squared Error for each ticker:")
            for name in self.tickers.groups.keys():
                print(f"{name}: Train MSE = {self.train_errors[name][model]:.4f}, Valid MSE = {self.valid_errors[name][model]:.4f}, Test MSE = {self.test_errors[name][model]:.4f}")
        
        except:
            raise TypeError("model not trained")
        
    def y_predict_rescaled(self,
        model,
        name,
        n_test                   
    ):
        try:
            self.test_pred[name][model] = self.scalers[name].inverse_transform(self.test_pred[name][model]).flatten()
            self.y_pred_prc[name][model] = [np.exp(self.test_pred[name][model][i])*self.prc[name][-n_test:][i] for i in range(n_test)]
            self.y_test_prc[name][model] = self.prc[name][-n_test:]
        except:
            raise TypeError("model not trained")