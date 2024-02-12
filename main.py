import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
from sklearn.model_selection import  train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM

today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = today - timedelta(days=5000)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download('TSLA',
                   start=start_date,
                   end=end_date,
                   progress=False)

data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
data.tail()

figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                        open=data["Open"],
                                        high=data["High"],
                                        low=data["Low"],
                                        close=data["Close"])])
figure.update_layout(title = "Tesla Stock Price Analysis",
                     xaxis_rangeslider_visible=False)
figure.show()


correlation = data.corr(numeric_only=True)
print("Correlations of features with the Close value (target):")
print(correlation["Close"].sort_values(ascending=False))



# split the data
x = data[["Open", "High", "Low", "Volume"]]
y = data[["Close"]]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1,1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)


# train an LSTM model for prediction
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=30)

# test the model
features = np.array([[186.77, 187.78, 186.56, 968800]])
print(model.predict(features))