# **Stock Market Predictor**
>**This projects was part of my graduation's final year project. It is a web application that can predict the upcomming stock market trend using RNN-LSTM neural network and advises wether to buy or sell stocks from the specific company.**

<img src="https://github.com/tionx3na/RNN-LSTM-Stock-Market-Trend-Predictor/blob/master/IMAGES/Screenshot%20(1).png">

### Problem Statement
> “The Stock Market is chaotic and unpredictable where any slight fluctuation in supply and demand will affect stock prices. The future stock market prices are therefore ambiguous and highly volatile and there is a need for predicting the future stock prices and fluctuations accurately.”

### Proposed Solution
> “To Create an interface which can help investors or financial advisors to predict stock prices which in turn gives them a competitive edge.”

## Control Flow 
<img src="https://github.com/tionx3na/RNN-LSTM-Stock-Market-Trend-Predictor/blob/master/IMAGES/flow.png">

* The user enters a company name. The company name is converted into the corresponding ticker symbol. 
* Historic stock quotes of the company are pulled from Yahoo Finance. Technical, Fundamental and Sentiment Analysis is performed on the dataset.
* The dataset is trained with an LSTM-RNN. 
* The Predicted values are shown as graph in a web based UI for the users. A recommendation of either buy or sell is given to the users along with other details.
