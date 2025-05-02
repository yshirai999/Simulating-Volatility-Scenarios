# Forecasting Arbitrage-Free Volatility Surfaces

- The purpose of this research is to analyse the time series of Bilateral Gamma parameters daily calibrated to options prices.

- Specifically, we aim at predicting (bp,cp,bn,cn), which correspond to the shape and scale parameters for the gamma process of gains and of losses respectively.

## Data format

- The k-th input data file must be a .mat file named BGPk and in the folder Data1

- It must be composed of two structs

- The first is composed of ticker names

- The second one of data in the following format: date, bp, cp, bn, cn, MSE from estimation

- An example containing SPY and 11 sector ETFs is in the Data1 folder

- Scale parameters are quantized by default

## Classes

- The classes in the python files data.py and BaseEnv.py read and process the data in the BGP.mat file

- The RegressionClass and RNNClarr inherit the properties and methods of the BaseClass, which in turn inherits from the dataclass

- The models in the RegressionClass currently supported are Linear Regression and Huber Regression

- Those in the RNNClass are SimpleRNN and LSTM

- The number of recurrent layers can be specified at the time an instance of the respective classes is constructed

## Outputs

- Illustrative examples for RNN and Regression are in the juptyer notebooks RNN.ipynb and Regression.ipynb

- Data may also be visualized in the DataPlots.ipynb notebook.
