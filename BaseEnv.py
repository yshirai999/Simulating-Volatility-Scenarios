import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
from data import dataclass
from typing import List
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class BaseClass(dataclass):

    def __init__(self,
    tickers: List[str] = ['spy'],
    feature_steps: int = 10,
    target_steps: int = 1,
    quantized: bool = True,
    quant_all:bool = False,
    nclusters: int = 525,
    scaler = StandardScaler,
    vars=[0,2]
    ):
        super().__init__()
        self.tickers = tickers
        self.BG = dict()
        self.dates = dict()
        self.df = dict()
        if quantized:
            self.BGPquant = self.quantization(nclusters, quant_all=quant_all)
        for t in self.tickers:
            loc = self.tickersloc[t]
            if quantized:
                BG = self.BGPquant[loc[0]]["parms"][self.BGP[loc[0]]["ticker"][loc[1]]]
            else:
                BG = self.BGP[loc[0]]["parms"][self.BGP[loc[0]]["ticker"][loc[1]]]
            d = str(int(BG[0][0]))
            self.BG[t] = BG[:][1:-1]
            self.df[t] = pd.DataFrame(np.transpose(self.BG[t]), columns = ["bp","cp","bn","cn"])
            self.dates[t] = [str(int(d)) for d in BG[:][0]]
            self.dates[t] = [dt.strptime(d, '%Y%m%d') for d in self.dates[t]]
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
        self.scaler = {}

        for t in self.tickers:
            data = self.df[t].diff().dropna()
            print(data.values.shape)
            self.ts[t] = data.values[:,vars] # change this line
            self.scaler[t] = scaler()
            self.scaler[t].fit(self.ts[t])
            self.ts[t] = self.scaler[t].transform(self.ts[t])
            self.X[t], self.y[t] = self.ts_split(self.ts[t])
            self.split_ind[t] = int(self.X[t].shape[0]*0.8)
            self.X_train_full[t], self.y_train_full[t] = self.X[t][:self.split_ind[t]], self.y[t][:self.split_ind[t]]
            self.X_test[t], self.y_test[t] = self.X[t][self.split_ind[t]:], self.y[t][self.split_ind[t]:]
            self.split_ind_2[t] = int(self.X_train_full[t].shape[0]*0.8)
            self.X_train[t], self.y_train[t] = self.X_train_full[t][:self.split_ind_2[t]], self.y_train_full[t][:self.split_ind_2[t]]
            self.X_valid[t], self.y_valid[t] = self.X_train_full[t][self.split_ind_2[t]:], self.y_train_full[t][self.split_ind_2[t]:]
    
    def ts_split(self,
        ts
    ):
        print(f"ts {ts.shape}")
        feature_steps = self.feature_steps
        target_steps = self.target_steps
        n_obs = len(ts) - feature_steps - target_steps + 1
        X = np.array([ts[idx:idx + feature_steps] for idx in range(n_obs)])
        y = np.array([ts[idx + feature_steps:idx + feature_steps + target_steps][-1,:]
                    for idx in range(n_obs)])
        return X, y
    
    def visualization_bVSc(self,
        tickers
    ):
        bp = dict()
        cp = dict()
        bn = dict()
        cn = dict()
        bp["all"] = np.array([])
        cp["all"] = np.array([])
        bn["all"] = np.array([])
        cn["all"] = np.array([])
        
        T = len(tickers)

        for t in range(T):
            fig = plt.figure()

            ax = fig.add_subplot(1,2,1)
            bp[t] = self.df[tickers[t]]["bp"].values
            cp[t] = self.df[tickers[t]]["cp"].values
            bn[t] = self.df[tickers[t]]["bn"].values
            cn[t] = self.df[tickers[t]]["cn"].values

            ax.scatter(bp[t], cp[t],s=1)
            ax.set_xlabel('bp')
            ax.set_ylabel('cp')
            ax.set_title(tickers[t]+': Positive jumps')

            ax = fig.add_subplot(1,2,2)
            ax.scatter(bn[t], cn[t],s=1)
            ax.set_xlabel('bn')
            ax.set_ylabel('cn')
            ax.set_title(tickers[t]+': Negative jumps')

            plt.tight_layout()
            plt.show()

            bp["all"] = np.concatenate([bp["all"],bp[t]])
            cp["all"] = np.concatenate([cp["all"],cp[t]])
            bn["all"] = np.concatenate([bn["all"],bn[t]])
            cn["all"] = np.concatenate([cn["all"],cn[t]])
        
        if T > 1:
            fig = plt.figure()
            ax = fig.add_subplot(1,2,1)
            ax.scatter(bp["all"], cp["all"],s=1)
            ax.set_xlabel('bp')
            ax.set_ylabel('cp')
            ax.set_title('All tickers: Positive jumps')

            ax = fig.add_subplot(1,2,2)
            ax.scatter(bp["all"], cp["all"],s=1)
            ax.set_xlabel('bp')
            ax.set_ylabel('cp')
            ax.set_title('All tickers: Positive jumps')

            plt.tight_layout()
            plt.show()




    def visualization_bVSc_3D(self,
        tickers
    ):
        bp = dict()
        cp = dict()
        bn = dict()
        cn = dict()
        bp["all"] = np.array([])
        cp["all"] = np.array([])
        bn["all"] = np.array([])
        cn["all"] = np.array([])
        
        T = len(tickers)

        for t in range(T):
            bp[t] = self.df[tickers[t]]["bp"].values
            cp[t] = self.df[tickers[t]]["cp"].values
            bn[t] = self.df[tickers[t]]["bn"].values
            cn[t] = self.df[tickers[t]]["cn"].values

            fig = plt.figure()

            ax = fig.add_subplot(1,2,1,projection='3d')
            ax.scatter(bp[t],bn[t],cp[t],s=1)
            ax.set_xlabel('bp')
            ax.set_ylabel('bn')
            ax.set_title(tickers[t]+': Positive jumps')

            ax = fig.add_subplot(1,2,2,projection='3d')
            ax.scatter(bp[t],bn[t],cn[t], s=1)
            ax.set_xlabel('bp')
            ax.set_ylabel('bn')
            ax.set_title(tickers[t]+': Negative jumps')

            plt.tight_layout()
            plt.show()

            bp["all"] = np.concatenate([bp["all"],bp[t]])
            cp["all"] = np.concatenate([cp["all"],cp[t]])
            bn["all"] = np.concatenate([bn["all"],bn[t]])
            cn["all"] = np.concatenate([cn["all"],cn[t]])
    
        if T > 1:
            fig = plt.figure()

            ax = fig.add_subplot(1,2,1,projection='3d')
            ax.scatter(bp['all'],bn['all'],cp['all'],s=1)
            ax.set_xlabel('bp')
            ax.set_ylabel('bn')
            ax.set_title('All: Positive jumps')

            ax = fig.add_subplot(1,2,2,projection='3d')
            ax.scatter(bp['all'],bn['all'],cn['all'],s=1)
            ax.set_xlabel('bp')
            ax.set_ylabel('bn')
            ax.set_title('All: Negative jumps')
        

    def visualization_ts_bc(self,
        tickers
    ):
        ts = dict()        
        var = ["bp", "cp", "bn", "cn"]
        T = len(tickers)

        for t in range(T):
            ts[t] = []
            fig, axes = plt.subplots(1, 4, figsize=(15, 5))
            for i in range(4):
                ts[t].append(self.df[tickers[t]][var[i]].values)
                axes[i].plot(self.dates[tickers[t]],ts[t][i])
                axes[i].set_xlabel('date')
                axes[i].set_ylabel(var[i])
                axes[i].set_title(tickers[t]+': '+var[i])
                
            plt.tight_layout()
            plt.show()