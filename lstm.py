import datetime
import backtrader as bt
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

class LSTMStrategy(bt.Strategy):
    params = (
        ('lookback', 10),
        ('batch_size', 8),
        ('epochs', 20),
        ('neurons', 30),
        ('verbose', False),
        ('threshold', 0.01),
        ('stop_loss', 0.6),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataclose = self.datas[0].close
        self.model = self.build_model()
        self.lookback = self.params.lookback
        self.batch_size = self.params.batch_size
        self.epochs = self.params.epochs
        self.neurons = self.params.neurons
        self.verbose = self.params.verbose
        self.threshold = self.params.threshold
        self.stop_loss = self.params.stop_loss
        self.history = [] # Initialize the history list

        # Initialize last_prediction after `lookback` iterations
        self.last_prediction = None
        if len(self) > self.lookback:
            x = np.array([self.history[-self.lookback:]]) # use the last `lookback` values
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))
            self.last_prediction = self.model.predict(x)[0][0]


    def build_model(self):
        model = Sequential()
        model.add(LSTM(self.params.neurons, input_shape=(self.params.lookback, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    def train_model(self, x_train, y_train):
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)


    def next(self):
        # current close price
        current_price = self.dataclose[0]

        # append the current price to the history
        self.history.append(current_price)

        # check if we have enough historical data
        if len(self.history) > self.lookback:
            # prepare the input data
            x = np.array([self.history[-self.lookback:]]) # use the last `lookback` values
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))

            # make a prediction
            prediction = self.model.predict(x)

            # train the model on historical data
            y = np.array([self.history[-self.lookback+1:]]) # use the last `lookback-1` values as targets
            y = np.reshape(y, (y.shape[0], y.shape[1], 1))
            self.train_model(x, y)

            # place a buy order if the predicted price is higher than the current price or if rsi is low
            if prediction[0][0] > current_price * self.threshold: # use threshold
                self.buy()

            # place a sell order if the current price is lower than the stop loss or if rsi is high
            elif current_price < self.position.price * self.stop_loss: # use stop loss
                self.close()

    def stop(self):
        # print the final portfolio value
        print('Final Portfolio Value: %.2f' % self.broker.getvalue())

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        if self.verbose:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))


if __name__ == '__main__':
    # Download stock data from Yahoo Finance
    omxs30 = yf.download('^OMX', start='2018-01-01', end='2019-01-01')
    omxs30.to_csv('omxs30.csv')

    # Create a Cerebro instance
    cerebro = bt.Cerebro()

    # Add the data to Cerebro
    data = bt.feeds.GenericCSVData(
        dataname='omxs30.csv',
        datetime=0,
        dtformat='%Y-%m-%d', # Update datetime format
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1

    )
    cerebro.adddata(data)

    # Add the LSTM strategy to Cerebro
    cerebro.addstrategy(LSTMStrategy)
    # Run the backtest
    cerebro.run()

    cerebro.plot()
    # Create a new plot with only the data we want to show
    # Save the plot to a file
    plt.savefig('plot3.png')



