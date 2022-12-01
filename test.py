import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import datetime as dt
from scipy.stats import norm
from models import multivariate_var
from metrics import pof_test, if_test, quantile_loss

class TestVaR:
    def test_VaR(self):

        tickers_list = [ ['AAPL', 'C', 'TSLA'],['DIS','TSLA','ADSK','AMC'],
                         ['AAPL','BABA', 'C', 'DIS','TSLA','ADSK','AMC'] ]
        weights_list = [np.array([0.1, 0.1, 0.8]), np.array([0.05, 0.05, 0.4, 0.05]),
                        np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05])]
        from_date_list = ['2018-02-01', '2018-04-01', '2018-08-01']
        to_date_list = ['2020-05-01', '2020-08-01', '2021-01-01']
        intitial_investment_list = [1000,1050,500]
        alpha_list = [0.01, 0.05, 0.07]
        n_list = [10, 50, 100]
        results = [259.41,157.75,144.39]

        for values, result in zip(zip(tickers_list, weights_list, from_date_list, to_date_list,
                                      intitial_investment_list, alpha_list, n_list), results):
            assert (multivariate_var(*values) == result )


class TestMetrics:
    def test_quantile_loss(self):
        np.random.seed(0)
        target = np.random.randn(10)
        var = np.ones(10)
        assert np.isclose(quantile_loss(var, target, alpha=0.9), 1.0461, atol=0.0001)
        assert np.isclose(quantile_loss(var, target, alpha=0.1), 0.6269, atol=0.0001)

    def test_pof_test(self):
        np.random.seed(0)
        target = np.random.randn(10)
        var = -np.ones(10) * 2
        assert np.isclose(pof_test(var, target, alpha=0.95), 0.3111, atol=0.0001)

    def test_if_test(self):
        np.random.seed(0)
        target = np.random.randn(1000)
        var = -np.ones(1000) * 2
        assert np.isclose(if_test(var, target), 0.2770, atol=0.0001)


