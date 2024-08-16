import requests, time, re, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

"""
author - Kaneel Senevirathne
date - 1/8/2022
stock utils for preparing training data.
"""

# Seeking Alpha API
SEEKING_ALPHA_API_KEY = 'YOUR_SEEKING_ALPHA_API_KEY'



def timestamp(dt):
    epoch = datetime.utcfromtimestamp(0)
    return int((dt - epoch).total_seconds() * 1000)


def linear_regression(x, y):
    """
    performs linear regression given x and y. outputs regression coefficient
    """
    #fit linear regression
    lr = LinearRegression()
    lr.fit(x, y)
    
    return lr.coef_[0][0]

def n_day_regression(n, df, idxs):
    """
    n day regression.
    """
    #variable
    _varname_ = f'{n}_reg'
    df[_varname_] = np.nan

    for idx in idxs:
        if idx > n:
            
            y = df['close'][idx - n: idx].to_numpy()
            x = np.arange(0, n)
            #reshape
            y = y.reshape(y.shape[0], 1)
            x = x.reshape(x.shape[0], 1)
            #calculate regression coefficient 
            coef = linear_regression(x, y)
            df.loc[idx, _varname_] = coef #add the new value
            
    return df

def normalized_values(high, low, close):
    """
    normalize the price between 0 and 1.
    """
    #epsilon to avoid deletion by 0
    epsilon = 10e-10
    
    #subtract the lows
    high = high - low
    close = close - low
    return close/(high + epsilon)

def get_stock_price(stock, date):
    """
    returns the stock price given a date
    """
    start_date = date - timedelta(days = 10)
    end_date = date
    
    #enter url of database
    url = f"https://seekingalpha.com/api/v3/market_data/{stock}/historical_prices"
    headers = {
        "Authorization": f"Bearer {SEEKING_ALPHA_API_KEY}"
    }
    query = {
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
    }

    response = requests.get(url, headers=headers, params=query)
    data = response.json()
    
    
    try:
        data = pd.DataFrame(data['prices'])
        data['date'] = pd.to_datetime(data['date'])
        return data['close'].values[-1]
    except:
        pass

def get_data(sym, start_date=None, end_date=None, n=10):
    url = f"https://seekingalpha.com/api/v3/market_data/{sym}/historical_prices"
    headers = {
        "Authorization": f"Bearer {SEEKING_ALPHA_API_KEY}"
    }
    
    if start_date:
        payload = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
        }
    else:
        payload = {
            'start_date': '2007-01-01',
            'end_date': '2020-12-31',
        }
            
    response = requests.get(url, headers=headers, params=payload)
    data = response.json()
    
    data = pd.DataFrame(data['prices'])
    data['date'] = pd.to_datetime(data['date'])

    data['normalized_value'] = data.apply(lambda x: normalized_values(x.high, x.low, x.close), axis=1)
    data['loc_min'] = data.iloc[argrelextrema(data.close.values, np.less_equal, order=n)[0]]['close']
    data['loc_max'] = data.iloc[argrelextrema(data.close.values, np.greater_equal, order=n)[0]]['close']

    idx_with_mins = np.where(data['loc_min'] > 0)[0]
    idx_with_maxs = np.where(data['loc_max'] > 0)[0]
    
    return data, idx_with_mins, idx_with_maxs

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
