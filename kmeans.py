import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import backtrader as bt
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class StockData(bt.feeds.GenericCSVData):
    params = (
        ('nullvalue', float('nan')),
        ('dtformat', ('%Y-%m-%d')),
        ('tmformat', ('%H:%M:%S')),
        ('datetime', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', -1)
    )

class KMeansStrategy(bt.Strategy):
    params = (
        ('n_clusters', 3),
        ('max_iter', 100),
        ('preprocessing', None),
        ('n_init', 10)
    )

    def __init__(self):
        self.data_clustered = False

        # Define k-means clustering algorithm
        self.kmeans = cluster.KMeans(
            n_clusters=self.params.n_clusters,
            max_iter=self.params.max_iter,
            n_init=self.params.n_init
            
            
        )

    def preprocess_data(self, data):
        # Convert data to pandas dataframe
        data = pd.DataFrame(data, columns=['close'])

        # Apply any preprocessing steps here
        if self.params.preprocessing == 'log':
            data = np.log(data)
        elif self.params.preprocessing == 'diff':
            data = np.diff(data)
        elif self.params.preprocessing == 'pct_change':
            data = data.pct_change().dropna()

        return data

    def next(self):
        if not self.data_clustered:
            # Preprocess data
            data = self.preprocess_data(self.data.close.get(size=self.params.max_iter))

            # Only proceed with clustering if data contains at least one sample
            if len(data) > 0:
                # Apply k-means clustering
                self.kmeans.fit(data)

                # Use resulting clusters to make trading decisions
                labels = self.kmeans.labels_
                cluster_counts = np.bincount(labels)
                dominant_cluster = np.argmax(cluster_counts)
                if labels[-1] == dominant_cluster:
                    self.buy()

                self.data_clustered = True



if __name__ == '__main__':
    omxs30 = yf.download('^OMX', start='2010-01-01', end='2022-01-01')
    omxs30.to_csv('omxs30.csv')
    cerebro = bt.Cerebro()
    cerebro.addstrategy(KMeansStrategy, n_clusters=3, max_iter=100, preprocessing='pct_change')
    data = StockData(dataname='omxs30.csv')
    cerebro.adddata(data)
    cerebro.run()

    cerebro.plot()
    plt.savefig('plot.png')
