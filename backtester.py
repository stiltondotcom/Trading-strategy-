"""
stock backtester to test the model given a dataset. 
author - Kaneel Senevirathne
date - 1/13/2022
"""

import requests, time, re, os
import matplotlib.pyplot as plt

import numpy as np
from stock_utils.simulator import simulator
from stock_utils.stock_utils import get_stock_price
from models import lr_inference
from datetime import datetime
from datetime import timedelta

import pandas as pd

from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from models.lr_inference import LR_v1_predict, LR_v1_sell
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")
import os
import pickle
from tqdm import tqdm

# Seeking Alpha API - Replace with your actual API key
SA_API_KEY = 'XXXXX' ### your Seeking Alpha API key

def timestamp(dt):
    epoch = datetime.utcfromtimestamp(0)
    return int((dt - epoch).total_seconds() * 1000)

def linear_regression(x, y):
    """
    performs linear regression given x and y. outputs regression coefficient
    """
    lr = LinearRegression()
    lr.fit(x, y)
    
    return lr.coef_[0][0]

def n_day_regression(n, df, idxs):
    """
    n day regression.
    """
    _varname_ = f'{n}_reg'
    df[_varname_] = np.nan

    for idx in idxs:
        if idx > n:
            y = df['close'][idx - n: idx].to_numpy()
            x = np.arange(0, n)
            y = y.reshape(y.shape[0], 1)
            x = x.reshape(x.shape[0], 1)
            coef = linear_regression(x, y)
            df.loc[idx, _varname_] = coef
            
    return df

def normalized_values(high, low, close):
    """
    normalize the price between 0 and 1.
    """
    epsilon = 10e-10
    high = high - low
    close = close - low
    return close/(high + epsilon)

def get_stock_price(stock, date):
    """
    returns the stock price given a date using Seeking Alpha API
    """
    start_date = (date - timedelta(days=10)).strftime('%Y-%m-%d')
    end_date = date.strftime('%Y-%m-%d')
    
    url = f'https://seeking-alpha.p.rapidapi.com/market/get-chart'
    headers = {
        'X-RapidAPI-Key': SA_API_KEY,
        'X-RapidAPI-Host': 'seeking-alpha.p.rapidapi.com'
    }
    query = {
        'symbol': stock,
        'period': '10d',
        'interval': '1d'
    }

    results = requests.get(url, headers=headers, params=query)
    data = results.json()
    
    try:
        # Assuming the Seeking Alpha API returns similar structured data
        prices = [day['close'] for day in data['data']]
        return prices[-1]
    except:
        pass
    
def get_data(sym, start_date=None, end_date=None, n=10):
    """
    Fetch historical stock data using Seeking Alpha API
    """
    if not start_date:
        start_date = datetime(2007, 1, 1)
    if not end_date:
        end_date = datetime(2020, 12, 31)

    url = f'https://seeking-alpha.p.rapidapi.com/market/get-chart'
    headers = {
        'X-RapidAPI-Key': SA_API_KEY,
        'X-RapidAPI-Host': 'seeking-alpha.p.rapidapi.com'
    }
    query = {
        'symbol': sym,
        'period': '10y',  # Adjust as needed
        'interval': '1d',
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d')
    }

    results = requests.get(url, headers=headers, params=query)
    data = results.json()
    
    # Assuming the Seeking Alpha API returns similar structured data
    df = pd.DataFrame(data['data'])
    df['date'] = pd.to_datetime(df['date'])

    df['normalized_value'] = df.apply(lambda x: normalized_values(x.high, x.low, x.close), axis=1)
    df['loc_min'] = df.iloc[argrelextrema(df.close.values, np.less_equal, order=n)[0]]['close']
    df['loc_max'] = df.iloc[argrelextrema(df.close.values, np.greater_equal, order=n)[0]]['close']

    idx_with_mins = np.where(df['loc_min'] > 0)[0]
    idx_with_maxs = np.where(df['loc_max'] > 0)[0]
    
    return df, idx_with_mins, idx_with_maxs

def create_train_data(stock, start_date=None, end_date=None, n=10):
    data, idxs_with_mins, idxs_with_maxs = get_data(stock, start_date, end_date, n)
    
    data = n_day_regression(3, data, list(idxs_with_mins) + list(idxs_with_maxs))
    data = n_day_regression(5, data, list(idxs_with_mins) + list(idxs_with_maxs))
    data = n_day_regression(10, data, list(idxs_with_mins) + list(idxs_with_maxs))
    data = n_day_regression(20, data, list(idxs_with_mins) + list(idxs_with_maxs))
  
    _data_ = data[(data['loc_min'] > 0) | (data['loc_max'] > 0)].reset_index(drop=True)
    _data_['target'] = [1 if x > 0 else 0 for x in _data_.loc_max]
    
    cols_of_interest = ['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg', 'target']
    _data_ = _data_[cols_of_interest]
    
    return _data_.dropna(axis=0)

def create_test_data_lr(stock, start_date=None, end_date=None, n=10):
    data, _, _ = get_data(stock, start_date, end_date, n)
    idxs = np.arange(0, len(data))
    
    data = n_day_regression(3, data, idxs)
    data = n_day_regression(5, data, idxs)
    data = n_day_regression(10, data, idxs)
    data = n_day_regression(20, data, idxs)
    
    cols = ['close', 'volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']
    data = data[cols]

    return data.dropna(axis=0)

def predict_trend(stock, _model_, start_date=None, end_date=None, n=10):
    data, _, _ = get_data(stock, start_date, end_date, n)
    
    idxs = np.arange(0, len(data))
    data = n_day_regression(3, data, idxs)
    data = n_day_regression(5, data, idxs)
    data = n_day_regression(10, data, idxs)
    data = n_day_regression(20, data, idxs)
        
    data['pred'] = np.nan

    cols = ['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']
    x = data[cols]

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    for i in range(x.shape[0]):
        try:
            data['pred'][i] = _model_.predict(x[i, :])
        except:
            data['pred'][i] = np.nan

    return data



class backtester(simulator):

    def __init__(self, stocks_list, model, capital, start_date, end_date, threshold = 0.99, sell_perc = 0.04, hold_till = 5,\
         stop_perc = 0.005):
        
        super().__init__(capital) #initialize simulator

        self.stocks = stocks_list
        self.model = model
        self.start_date = start_date
        self.day = start_date
        self.end_date = end_date  
        self.status = 'buy' #the status says if the backtester is in buy mode or sell mode
        self.threshold = threshold
        self.sell_perc = sell_perc
        self.hold_till = hold_till
        self.stop_perc = stop_perc

        #current directory
        current_dir = os.getcwd()
        results_dir = os.path.join(current_dir, 'results')
        folder_name = f'{str(self.model.__name__)}_{self.threshold}_{self.hold_till}'
        self.folder_dir = os.path.join(results_dir, folder_name)
        if not os.path.exists(self.folder_dir):
            #create a new folder
            os.makedirs(self.folder_dir)
      
    def backtest(self):
        """
        start backtesting
        """
        delta = timedelta(days = 1)
        
        #progress bar to track progress
        total_days = (self.end_date - self.start_date).days
        d = 0
        pbar = tqdm(desc = 'Progress', total = total_days)

        while self.day <= self.end_date:
            
            #daily scanner dict
            self.daily_scanner = {}  
            if self.status == 'buy':
                #scan stocks for the day
                self.scanner()
                if list(self.daily_scanner.keys()) != []:
                    recommended_stock = list(self.daily_scanner.keys())[0]
                    recommended_price = list(self.daily_scanner.values())[0][2]
                    self.buy(recommended_stock, recommended_price, self.day) #buy stock
                    # print(f'Bought {recommended_stock} for {recommended_price} on the {self.day}')
                    self.status = 'sell' #change the status to sell
                else:
                    # print('No recommendations')
                    pass
            else: #if the status is sell
                #get stock price on the day
                stocks = [key for key in self.buy_orders.keys()]
                for s in stocks:
                    recommended_action, current_price = LR_v1_sell(s, self.buy_orders[s][3], self.buy_orders[s][0], self.day, \
                        self.sell_perc, self.hold_till, self.stop_perc)
                    if recommended_action == "SELL":
                        # print(f'Sold {s} for {current_price} on {self.day}')
                        self.sell(s, current_price, self.buy_orders[s][1], self.day)
                        self.status = 'buy'              
            #go to next day
            self.day += delta
            d += 1
            pbar.update(1)
        pbar.close()
        #sell the final stock and print final capital also print stock history 
        self.print_bag()
        self.print_summary() 
        self.save_results()      
        return

    def get_stock_data(self, stock, back_to = 40):
        """
        this function queries to td database and get data of a particular stock on a given day back to certain amount of days
        (default is 30). 
        """
        #get start and end dates
        end = self.day
        start = self.day - timedelta(days = back_to)        
        # prediction, prediction_thresholded, close_price = LR_v1_predict(stock, start, end, threshold = 0.5)
        prediction, prediction_thresholded, close_price = self.model(stock, start, end, self.threshold)
        return prediction[0], prediction_thresholded, close_price

    def scanner(self):
        """
        scan the stocks to find good stocks
        """
        for stock in self.stocks:
            try:#to ignore the stock if no data is available. #for staturdays or sundays etc
                prediction, prediction_thresholded, close_price = self.get_stock_data(stock)
                #if prediction greater than
                if prediction_thresholded < 1: #if prediction is zero
                    self.daily_scanner[stock] = (prediction, prediction_thresholded, close_price)
            except:
                pass

        def take_first(elem):
            return elem[1]

        self.daily_scanner = OrderedDict(sorted(self.daily_scanner.items(), key = take_first, reverse = True))

    def save_results(self):
        """
        save history dataframe create figures and save
        """
        #save csv file
        results_df_path = os.path.join(self.folder_dir, 'history_df.csv')
        self.history_df.to_csv(results_df_path, index = False)
        
        #save params and results summary
        results_summary_path = os.path.join(self.folder_dir, 'results_summary')
        results_summary = [self.initial_capital, self.total_gain]
        params_path = os.path.join(self.folder_dir, 'params')
        params = [self.threshold, self.hold_till, self.sell_perc, self.stop_perc, self.start_date, self.end_date]
        
        with open(results_summary_path, 'wb') as fp:
            pickle.dump(results_summary, fp)
        with open(params_path, 'wb') as fp:
            pickle.dump(params, fp)

if __name__ == "__main__":
    #stocks list
    dow = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC',\
        'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH',\
        'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS']
    other = ['AMD', 'MU', 'ABT', 'AAL', 'UAL', 'DAL', 'ANTM', 'ATVI', 'BAC', 'PNC', 'C', 'EBAY', 'AMZN', 'GOOG', 'FB', 'SNAP', 'TWTR'\
        'FDX', 'MCD', 'PEP', ]

    stocks = list(np.unique(dow + other))
    back = backtester(dow, LR_v1_predict, 3000, datetime(2021, 1, 1), datetime(2021, 12, 31), threshold = 0.9, sell_perc = 0.03, hold_till = 10,\
        stop_perc = 0.03)
    back.backtest()

    


    
