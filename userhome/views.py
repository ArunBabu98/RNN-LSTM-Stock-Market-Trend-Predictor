from django.shortcuts import render
import yfinance as yf
import bs4
import requests
import pandas as pd
from bs4 import BeautifulSoup
from django.http import HttpResponse
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, CuDNNLSTM
from sklearn.metrics import roc_curve,roc_auc_score
import plotly.express as plt
import plotly.graph_objects as go
import numpy as np
import math
from textblob import TextBlob
import finviz
import nltk
import matplotlib as mplt
# Create your views here.


def userhome(request):
    return render(request,"userhome.html")

def scrape(request):
    # scraping for ticker name
    company= request.GET['company']
    r = requests.get('https://www.marketwatch.com/tools/quotes/lookup.asp?siteID=mktw&Lookup=' + company + '&Country=All&Type=All')
    soup = bs4.BeautifulSoup(r.text, "lxml")
    table = soup.find('div', class_='results')
    rows = table.find_all('td', class_='bottomborder')
    use= rows[0].text
    ticker = yf.Ticker(use)
    # converting into csv and json files
    hist = ticker.history(period="max")
    hist.to_csv('hist.csv')
    hist.to_json('hist.json')
    # Scraping details from Yahoo finance
    l = requests.get('https://in.finance.yahoo.com/quote/' + use + '?p=' + use + '&.tsrc=fin-srch')
    soup2 = bs4.BeautifulSoup(l.text, "lxml")
    div1 = soup2.find('div', class_='Mt(15px)').text
    div1_1 = soup2.find('div', class_='My(6px) Pos(r) smartphone_Mt(6px)').text
    # plotting graph of close price
    df = pd.read_csv('hist.csv',)
    df['movavg'] = df['Close'].rolling(window=30).mean()
    fig = plt.line(df, x='Date', y='Close', title='Detailed Graph', hover_name='Date')
    fig.add_scatter(x=df['Date'], y=df['movavg'], mode='lines', name='Moving Average')
    fig.add_scatter(x=df['Date'], y=df['Open'], mode='lines', name='Open')
    fig.add_scatter(x=df['Date'], y=df['High'], mode='lines', name='High')
    fig.add_scatter(x=df['Date'], y=df['Low'], mode='lines', name='Low')
    fig.add_scatter(x=df['Date'], y=df['Volume'], mode='lines', name='Volume', visible='legendonly')
    graph = fig.to_html(full_html=False)

    #Scraping for sentiment analysis
    # data preprocessing
    news = finviz.get_news(use)
    df = pd.DataFrame(news, columns=['text', 'link'])
    df.to_csv("News.csv")  # news data scraped and saved to csv file
    return render(request, "userhome.html", {'div1':div1, 'div2_1':div1_1,'graph': graph})

def predict(request):
    # ----------Data Pre processing ------------
    # read data from csv
    df = pd.read_csv('hist.csv', parse_dates=['Date'])
    # set date as index
    df.set_index('Date', inplace=True)
    df['movavg'] = df['Close'].rolling(window=30).mean()
    data = df.drop(columns=['Dividends', 'Stock Splits', 'Volume', 'Low', 'High', 'Open', 'movavg'])
    # end plotting

    # splitting the data into training and testing set

    x = data.values

    # Calculating the length for training data

    training_data_len = math.ceil(len(x) * .8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(x)

    # creating the scaled training data set

    # Create the scaled training data set
    train_data = scaled_data[0:training_data_len, :]

    # splitting the training data into x train and y train

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # -------------------end of data preprocessing------------------------

    # -------------------Building RNN------------------------------------

    # building an LSTM model
    model = Sequential()
    model.add(
        CuDNNLSTM(units=60, return_sequences=True, input_shape=(x_train.shape[1], 1)))  # ReLu Rectified Linear unit
    model.add(Dropout(0.2))

    model.add(CuDNNLSTM(units=60, return_sequences=True))  # ReLu Rectified Linear unit
    model.add(Dropout(0.3))

    model.add(CuDNNLSTM(units=80, return_sequences=True))  # ReLu Rectified Linear unit
    model.add(Dropout(0.4))

    model.add(CuDNNLSTM(units=120, return_sequences=False))  # ReLu Rectified Linear unit
    model.add(Dropout(0.5))

    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, batch_size=128, epochs=500)

    # ------------------------end of rnn modelling-----------------------------

    # --------------------prediction and visualisation------------------------

    # Create the testing data set

    # Create a new array containing scaled values
    test_data = scaled_data[training_data_len - 60:, :]

    # Create the data sets x_test and y_test
    x_test = []
    y_test = x[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)


    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))


    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    fig = plt.line(valid, x=valid.index, y='Close', title='Predicted')
    fig.add_scatter(x=df.index, y=df['Close'], mode='lines', name='Full Close Price')
    fig.add_scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Prediction')

    graph = fig.to_html(full_html=False)

    processing = " Stock Close Price Prediction Graph "

    last_day_len = len(predictions)
    last_day = predictions[last_day_len-1]

    #End of prediction model

    #Sentiment analysis of the company
    news = pd.read_csv('News.csv') #reading the csv file
    text = news["text"]
    text.replace("[^a-zA-Z]", " ", regex=True, inplace=True)  # Cleaned the text of any punctuation marks

    # Joining all the rows together into headlines[]
    ' '.join(str(x) for x in text.iloc[0:99])

    headlines = []
    for row in range(0, len(text.index)):
        headlines.append(' '.join(str(x) for x in text.iloc[0:99]))
    # end of joining all the rows

    # converting the headline into all lower case letters
    final_headline = [x.lower() for x in headlines]

    # Converting list into string
    new = " "

    for x in final_headline:
        new += x

    # Creating textblob object for sentiment analysis
    blob = TextBlob(new)

    # Converting polarity into 1,-1 or 0

    if blob.polarity > 0:
        value = 1
        reaction = "Positive"
    elif blob.polarity < 0:
        value = -1
        reaction = "Negative"
    elif blob.polarity == 0:
        value = 0
        reaction = "Neutral Sentiment"

    #Buy or sell analysis
    movavg = df['movavg']
    close = df['Close']

    movavg_tail = movavg.tail(1)

    close_tail = close.tail(1)

    x = movavg_tail.to_string(index=False)
    y = close_tail.to_string(index=False)

    if (y > x):
        result = "Buy"
        comment = "The Close Price is Higher than the moving average"

    else:
        result = "Sell"
        comment = "The close Price is Lower than the moving average"

    if (value == 1 and result == "Buy"):
        obj = "BUY"
    elif(value == 0 and result == "Buy"):
        obj = "BUY"
    elif(value == -1 and result == "BUY"):
        obj = "Sell"
    elif(value == 1 and result == "Sell"):
        obj = "Buy"
    elif(value == 0 and result == "Sell"):
        obj = "Sell"
    elif(value == -1 and result == "Sell"):
        obj = "Sell"
    else:
        obj = "Sell"

    return render(request, "userhome.html", {'graph':graph,'rmse': rmse, 'processing':processing, 'last_day':last_day, 'value':value, 'reaction':reaction, 'result':result, 'comment':comment,'obj':obj} )




