import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import datetime as dt
from scipy.stats import norm


def multivariate_var(tickers,  # list of tickers
                     weights,  # np.array of weights
                     from_date,  # date in format yyyy-mm-dd
                     to_date,  # date in format yyyy-mm-dd
                     initial_investment,  # value of initial investment (integer or float)
                     alpha,  # alpha, where 1 - alpha = confidence level
                     n  # number of days for n-days VaR calculation
                     ):
#создание и загрузка доходностей
    returns = pd.DataFrame(columns=tickers)
    for ticker in tickers:
        data = pdr.get_data_yahoo(ticker, from_date, to_date)

        data['Return'] = np.zeros(len(data))
        for i in range(1, len(data)):
            data['Return'][i] = ((data['Close'][i] - data['Close'][i - 1]) / data['Close'][i])
        returns[ticker] = data['Return']
    returns = returns.iloc[1:, :]
#среднее + матрица ковариаций доходностей
    mean = np.mean(returns.values, axis=0)
    cov = np.cov(returns.values, rowvar=False)

    u = np.dot(weights.transpose(), mean)
    sigma = np.dot(weights.transpose(), cov)
    sigma = np.dot(sigma, weights)
#ответ
    ret = norm.ppf(1 - alpha, loc=u, scale=(n * sigma) ** 0.5)
    return ret
