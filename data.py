import numpy as np
import h5py
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class dataclass:

    def __init__(self,
        folder = '//Simulating-Volatility-Scenarios//Data1//'
    ):
        self.path = os.path.dirname(os.getcwd()) 
        self.dir_path = self.path+folder
        dir_path = self.dir_path
        self.count = 0
        for path in os.listdir(dir_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)):
                self.count += 1
        count = self.count
        self.BGP = dict()
        self.tickersloc = dict()
        for k in range(1,count+1):
            self.BGP[k] = dict()
            filepath = f"{dir_path}BGP{k}.mat"
            f = h5py.File(filepath, "r")
            path=f"/d{k}r/"
            group = f[path]
            parms = group["parms"]
            ticker = group["ticker"]
            self.BGP[k]["ticker"] = list()
            self.BGP[k]["parms"] = dict()
            for t in ticker:
                st = t[0]
                obj = np.concatenate(f[st])
                str = "".join(["".join(chr(c)) for c in obj])
                self.BGP[k]["ticker"].append(str)
            count = 0
            for p in parms:
                sp = p[0]
                self.BGP[k]["parms"][self.BGP[k]["ticker"][count]] = np.array(f[sp])
                self.tickersloc[self.BGP[k]["ticker"][count]] = [k,count]
                count += 1
        self.BGPquant = dict()
        
    def quantization(self, n_clusters: int = 525) -> dict:
        Data1 = list()
        count = self.count
        for j in range(1,count+1):
            Data = list(self.BGP[j]['parms'].values())
            K = len(Data)
            for k in range(K):
                T = Data[k].shape[1]
                d = Data[k][1:-1].T
                for t in range(T):
                    Data1.append(d[t])
        datarray = np.array(Data1)

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(datarray[:,[0,2]])
        self.labels = self.kmeans.labels_
        self.centers = self.kmeans.cluster_centers_

        BGPquant = self.BGP
        for j in range(1,count+1):
            Data = list(self.BGP[j]['parms'].values())
            K = len(Data)
            for k in range(K):
                T = Data[k].shape[1]
                d = Data[k][[1,3]].T
                quantized = self.centers[self.kmeans.predict(d)]
                BGPquant[j]["parms"][self.BGP[j]["ticker"][k]][[1,3],:] = quantized.T
        plt.plot(BGPquant[1]["parms"]['aapl'][1,:])
        return BGPquant
