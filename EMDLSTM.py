
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from tensorflow import keras


class emdlstm():

    def __init__(self,startdate,enddate,seq_length):
        self.startdate = startdate
        self.enddate = enddate
        self.seq_length = seq_length

    """take in a time series and returns three decomposed series describing the former"""
    def stl(self,data):
        result = STL(data,period=2)
        result = result.fit()
        return result.trend, result.seasonal, result.resid

    """fetches the closing price data of the instrument in one day interval"""
    def get_data(self,ticker):
        return yf.download(ticker,start=self.startdate,end =self.enddate,interval='1d')['Close']

    """normalizes the fetched data"""
    def scaled(self,data):
        scaled = np.zeros(len(data))
        for i in range(len(data)):
            scaled[i] = (data[i] - data.min()) / (data.max() - data.min())
        return scaled

    """takes in a time series and retuns the sequences of data suitable for an RNN"""
    def sequences(self,data):
        X = []
        y = []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(data[i+self.seq_length])
        return np.array(X), np.array(y)

    """splits the imput data into training and test sets"""
    def split(self,X,y):
        X_train, X_test = X[:int(0.9 * len(X))], X[int(0.9 * len(X)):]
        y_train, y_test = y[:int(0.9 * len(X))], y[int(0.9 * len(X)):]

        return X_train, X_test, y_train, y_test

    """translates the normalized predicted values back to original scale"""
    def inverse(self,y,data):
        inv_scal = np.zeros(len(y))
        for i in range(len(y)):
            inv_scal[i] = y[i] * (data.max() - data.min()) + data.min()
        return inv_scal


start_date = '2014-06-30'
end_date = '2024-08-16'

instance = emdlstm(startdate=start_date,enddate=end_date,seq_length=60)  # initiate the class
dji = '^GSPC'                   # ticker symbol for s&p 500

data = instance.get_data(dji)     # fetch the closing prices
scaled_data = instance.scaled(data)    # normalize the series

trend, seasonal, resid = instance.stl(scaled_data)   # decompose the time series
trend_seq, _ = instance.sequences(trend)      # generate sequnces from the decomposed time series
seasonal_seq, _ = instance.sequences(seasonal)
resid_seq, _ = instance.sequences(resid)

X = np.stack((trend_seq,seasonal_seq,resid_seq),axis=-1)       # prepare the input suitable to train the model
_,y = instance.sequences(scaled_data)

X_train, X_test, y_train, y_test = instance.split(X,y)


"""build the CNN-LSTM network"""
model = keras.Sequential()
model.add(keras.layers.Conv1D(512,5))
model.add(keras.layers.MaxPool1D(2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(200,return_sequences=True,input_shape=(60,3)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64))
model.add(keras.layers.Dense(1))


model.compile('adam', loss='mean_squared_error')


model.fit(X_train,y_train,epochs=40,validation_split=0.2)    # train the model

mymodel = keras.models.load_model('s&p.h5')
y_pred = mymodel.predict(X_test)

y_pred = instance.inverse(y_pred,data)           # revert back to original scale
y_test = instance.inverse(y_test,data)
print(np.sqrt(np.mean((y_test-y_pred)**2)))
plt.plot(y_pred)
plt.plot(y_test)
plt.show()
mymodel.summary()
model.summary()
model.save('s&p.h5')

new_model = keras.models.load_model('s&p.h5')

infy = '^DJI'

new_data = instance.get_data(infy)
scaled_data_new = instance.scaled(new_data)

trend_new, seasonal_new, resid_new = instance.stl(scaled_data_new)
trend_seq_new, _ = instance.sequences(trend_new)
seasonal_seq_new, _ = instance.sequences(seasonal_new)
resid_seq_new, _ = instance.sequences(resid_new)

X_new = np.stack((trend_seq_new,seasonal_seq_new,resid_seq_new),axis=-1)
_,y_new = instance.sequences(scaled_data_new)



y_pred = new_model.predict(X_new)

plt.plot(scaled_data_new)
plt.plot(y_pred)
plt.show()

print(np.sqrt(np.mean((scaled_data_new-y_pred)**2)))

