import numpy as np
import h5py
import os

class dataclass:

    def __init__(self,
        folder = '//Simulating-Volatility-Scenarios//Data1//'
    ):
        self.path = os.path.dirname(os.getcwd()) 
        dir_path = self.path+folder
        count = 0
        for path in os.listdir(dir_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)):
                count += 1
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