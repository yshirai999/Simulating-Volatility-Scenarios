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

    def print_keys(self, obj, indent=0):
        if isinstance(obj, dict):
            for key, value in obj.items():
                print("  " * indent + str(key))
                self.print_keys(value, indent + 1)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                print("  " * indent + f"[{idx}]")
                self.print_keys(item, indent + 1)
        
    def quantization(self, n_clusters: int = 525, quant_all=False, vars=[1,3]) -> dict:
        Data1 = list()
        count = self.count
        for j in range(1,count+1):
            Data = list(self.BGP[j]['parms'].values())
            K = len(Data)
            for k in range(K):
                T = Data[k].shape[1] # 3273
                d = Data[k][1:-1].T # (3273, 4)
                for t in range(T):
                    Data1.append(d[t]) # d[t] is size 4
        datarray = np.array(Data1) # (601979, 4)
        # self.print_keys(self.BGP)
        print(datarray.shape)
        # Data is (46 tickers , 6 variable values, 3273 samples) 
        if (quant_all): # NEW: quantization that does 1d across all assets for each variable
            self.kmeans = {}
            self.centers = {}
            self.labels = {}
            print(count)
            for i in range(0,count):
                d = datarray[:,i].reshape(-1, 1)
                self.kmeans[i] = KMeans(n_clusters=n_clusters, random_state=0).fit(d)
                self.labels[i] = self.kmeans[i].labels_
                self.centers[i] = self.kmeans[i].cluster_centers_
                # AllQuantCenters[i] =  np.sort(self.kmeans.cluster_centers_.flatten())
            # print(AllQuantCenters)
            print(f"Quantized centers cp: {list(self.centers[1].flatten())}")
            # print(set(AllQuantCenters[0]))
            #return AllQuantCenters
            BGPquant = self.BGP
            for j in range(1,count+1):
                Data = list(self.BGP[j]['parms'].values())
                K = len(Data)
                for k in range(K):
                    T = Data[k].shape[1] # for each ticker in this list
                    # d = Data[k][[1,3]].T
                    d1 = Data[k][1].reshape(-1, 1) # get the ts for the values we want
                    d3 = Data[k][3].reshape(-1, 1) # get the ts for the values we want
                    quantized1 = self.centers[1][self.kmeans[1].predict(d1)] 
                    quantized3 = self.centers[3][self.kmeans[3].predict(d3)] 
                    BGPquant[j]["parms"][self.BGP[j]["ticker"][k]][1,:] = quantized1.T
                    BGPquant[j]["parms"][self.BGP[j]["ticker"][k]][3,:] = quantized3.T
        else:
            print("in here")
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(datarray[:,[0,2]])
            self.labels = self.kmeans.labels_
            self.centers = self.kmeans.cluster_centers_

            BGPquant = self.BGP
            for j in range(1,count+1):
                Data = list(self.BGP[j]['parms'].values())
                K = len(Data)
                for k in range(K):
                    # I think we can just change this code here to accept which values we want to quantize
                    T = Data[k].shape[1] # for each ticker in this list
                    d = Data[k][[0,2]].T # get the ts for the values we want
                    quantized = self.centers[self.kmeans.predict(d)] # quantize those separately
                    BGPquant[j]["parms"][self.BGP[j]["ticker"][k]][[0,2],:] = quantized.T
                
        plt.plot(BGPquant[1]["parms"]['aapl'][1,:])
        return BGPquant
