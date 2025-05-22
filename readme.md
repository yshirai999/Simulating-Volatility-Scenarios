# Simulating High Dimensional Volatility Scenarios

- The purpose of this research is to analyse the time series of Bilateral Gamma parameters daily calibrated to options prices.

- Specifically, we aim at predicting the next day bp and bn parameters, which correspond to the shape parameter for the gamma process of gains and of losses respectively

## Data format

- The k-th input data file must be a .mat file named BGPk and in the folder Data1

- It must be composed of two structs

- The first is composed of ticker names

- The second one of data in the following format: date, bp, cp, bn, cn, MSE from estimation

- An example containing SPY and 11 sector ETFs is in the Data1 folder

- Scale parameters are quantized by default

## Classes

- The classes in the python files data.py and BaseEnv.py read and process the data in the BGP.mat file

- The Transformer and Linear classes in Transformer.py and Linear.py inherit the properties and methods of the BaseClass, which in turn inherits from the dataclass

- The Transformer class creates an instance of the Transformer2D class, which implements a neural network with encoder-decoder layers and the self-attention mechanism

- The Linear class creates an instance of the Linear2D class, which implements a neural network with a single layer with no activation function

## Outputs

- Illustrative examples are in the juptyer notebooks Linear.ipynb and Transformer.ipynb

## Next steps

- Using softmax to assign likelyhood to next clusters rather than predicting it exactly
