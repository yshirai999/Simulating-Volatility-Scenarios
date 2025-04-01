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

- Try predicting the probabilities of each cluster for a small number (say 10 or 20) clusters created looking at the whole set of assets  
- This should be done for all the four parameters.
- It is essentially a classification problem, so the last layer activation function needs to be softmax
- The results should then be compared with standard hidden Markov modulation implementations (as done for instance in https://www.aimsciences.org/article/doi/10.3934/puqr.2022004)

- The question is: how do we train this network?
- Technically, we have several assets
- A naive way would be to fit them sequentially, until we observe some kind of convergence
- This could be limited to, say, SPY and the 9 ETFs, having in mind that they will be traded
- But, even simpler and more complex at the same time, we could even assume that the probabilities are different for each asset

- If we obtain a reliable fit, we could then implement a trading strategy for a single asset as follows:
- Call each cluster a positive one, if the mean log return bpcp-bncn > 0, and negative otherwise
- Then, we simply buy what is in positive signal state, and sell the negative ones
- We can do this every T days, where T is the expected time it takes for the majority of signals to switch sign
- The amount traded could be fixed for now, but we can also optimize it introducing a rebate

- Alternative, we could think of using this for hedging optimally
- If needed, we can then use Manifold learning to understand the intracluster noise geometry
