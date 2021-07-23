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

## Technical and Fundamental Analysis

1. Data Collection
> The historic stock quotes of a given company are web scraped through Yahoo Finance API and is saved as a CSV file in the local machine. The data necessary for company tickers are scraped from MarketWatch. The news journal data is scraped from finviz (Financial Visualizations).
2. Calculating technical trading signals
> Moving average is calculated from the complete close price over the years. We have used exponential moving average since it reacts more significantly to recent price changes than simple moving average. The exponential moving average is then integrated along with the existing CSV file. Price-to-Earning ratio is also calculated alongside and integrated into the CSV as well.
3. Numerical Representations
> We use min-max function to scale our data to values between 0 and 1 for computation. Then after training in neural network, inverse function is applied to get back the original values.
<img src="https://github.com/tionx3na/RNN-LSTM-Stock-Market-Trend-Predictor/blob/master/IMAGES/minmax.png">

## Sentiment Analysis
1. Data Collection and Preprocessing
>The data is web scraped from finviz where they provide news journals related to the specific companies. The scraped news is saved into a CSV file in the local machine. All the headlines are then joined together and the complete text is cleaned using a regular 
2. Data Analysis
> The cleaned data is then processed with TextBlob in order to get the polarity of the whole text. If the polarity is greater than 0 we set the sentiment as positive or else if the polarity is less than 0 then we set the sentiment as negative. The final sentiment is then integrated along with the result of the Technical Analysis and Fundamental Analysis.
<img src="https://github.com/tionx3na/RNN-LSTM-Stock-Market-Trend-Predictor/blob/master/IMAGES/textblob.png">

## Neural Network
>Our model uses a Long-Short Term Memory Recurrent Neural Network (LSTM-RNN). We chose this model because it is very efficient in time-series predictions. The vanishing gradient problems faced by RNN is completely avoided by the inclusion of the capability to remember the past data in LSTM. An LSTM node has 3 gates: Input Gate, Output Gate and Forget gate. Our models has an input layer which takes the prices of the first 60 days as the input. Our model has 3 hidden layers with drop out percentages of 30, 40 and 50 respectively. The output layer of our model predicts the price of 61st day. 80% of the dataset is for training and 20% is for testing. Our model has a batch size of 128 and epoch of 500. We use the Adam optimizer from Tensorflow which is an adaptive learning rate optimization algorithm with a default learning rate of  0.001

<img src="https://github.com/tionx3na/RNN-LSTM-Stock-Market-Trend-Predictor/blob/master/IMAGES/RNN%20LSTM.png">

## Technology Stacks used

<p float="left">
  <a href="https://www.djangoproject.com/"><img src="https://static.djangoproject.com/img/logos/django-logo-negative.png" height="100" width="200"></a>&nbsp;
  <a href="https://www.tensorflow.org/"><img src="https://www.tensorflow.org/images/tf_logo_social.png" height="100" width="200"></a> &nbsp;
  <a href="https://textblob.readthedocs.io/en/dev/"><img src="https://i.morioh.com/201014/76f74ea9.webp" height="100" width="200"></a> &nbsp;
  <a href="https://www.postgresql.org/"><img src="https://www.postgresql.org/media/img/about/press/slonik_with_black_text_and_tagline.gif" height="100" width="200"></a>
</p>

## Result
> The factors that are taken into account for the change in the closing price of a particular company are: prior closing price, open price, moving averages of the closing price and sentiment analysis of the company performance for the past days. We performed analysis on obtained data to establish a relation between our output parameters and the selected factors.
<img src="https://github.com/tionx3na/RNN-LSTM-Stock-Market-Trend-Predictor/blob/master/IMAGES/predicted.png">


## Market Result
> The final result of wether to buy the stock or sell the stock is shown in the web app after all the processing in the backend.
<img src="https://github.com/tionx3na/RNN-LSTM-Stock-Market-Trend-Predictor/blob/master/IMAGES/marketresult.png" width="200" height="200">

## Model Performance
> For Analysing the performance of our model we compared the model with other models using Weka and RapidMiner. We have found out that our model does comparatively better than the models on Weka and RapidMiner.
<img src="https://github.com/tionx3na/RNN-LSTM-Stock-Market-Trend-Predictor/blob/master/IMAGES/model%20performance.png">

## ROC Curve
> The ROC chart shows false positive rate on X-axis, and true positive rate on Y-axis. 
Our model was able to predict the analyst recommendation with an accuracy of above 78.4% which is the Area under ROC curve and the model was able to predict the close price with an RMSE value of 4.6
<img src="https://github.com/tionx3na/RNN-LSTM-Stock-Market-Trend-Predictor/blob/master/IMAGES/roc.png">

## vs Random Forest Classifier
>We trained our input dataset with a Random Forest Classifier in RapidMiner and we have found out that the RMS error is around 10.9 and the ROC curve significantly worse than our model.
<img src="https://github.com/tionx3na/RNN-LSTM-Stock-Market-Trend-Predictor/blob/master/IMAGES/randomforestclassifier.png">

- We also tested various other models on Weka like ADTree, NaïveBayes etc. where the ROC curve of our model is slightly better than these models and can predict the values more accurately than all of these models. 
- It is also noted that our model is slightly faster assuming all the models have the same batch size and epoch number.












