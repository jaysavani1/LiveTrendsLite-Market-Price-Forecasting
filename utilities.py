import os
import datetime as dt
from re import L
from dateutil.relativedelta import relativedelta as rtd

import pandas as pd
import numpy as np
from tqdm import tqdm

from yahooquery import Screener
import yfinance as yf

def getColNames():
    """_summary_

    Returns:
        _type_: dict
            returns the column names for data
    """
    
    mapper = {
        "Unnamed: 0":'Date',
        'Open':'Open',
        'High':'High',
        'Low':'Low',
        'Close':'Close',
        'Adj Close':'Adj Close',
        'Volume':'Volume',
    }
    return mapper

def getTimeDelta():
    """_summary_

    Returns:
        _type_: dict
            returns the granularity level pair with time delta
    """
    
    mapper = {
        "1m":-7, 
        "2m":-59, 
        "5m":-59, 
        "15m":-59, 
        "30m":-59,
        "90m":-59,
        "60m":-729,
        "1h":-729, 
        "1d":None,
        "5d":None,
        "1wk":None,
        "1mo":None, 
        "3mo":None,
        }
    return mapper

def getFolderNames():
    """_summary_

    Returns:
        _type_: dict
            retur the granularity level pair with folder names
    """
    mapper = {
        "1m":"1minute", 
        "2m":"2minute", 
        "5m":"5minute", 
        "15m":"15minute", 
        "30m":"30minute",
        "90m":"90minute", 
        "60m":"1hour",
        "1h":"1hour", 
        "1d":"daily",
        "5d":"5day",
        "1wk":"weekly",
        "1mo":"1month", 
        "3mo":"3month"
        }
    return mapper
    

def getTickers():
    """_summary_

    Returns:
        _type_: _description_
    """
    s = Screener()
    data = s.get_screeners('all_cryptocurrencies_us', count=250)

    # data is in the quotes key
    dicts = data['all_cryptocurrencies_us']['quotes']
    symbols = [d['symbol'] for d in dicts]
    
    return symbols

def getTickerData(ticker:str, start_date:str, end_date:str, interval:str):
    """_summary_

    Args:
        ticker (str): 
            Name of the coin. For Example: 'BTC-USD', 'ETH-USD','ADA-USD', etc.
        start_date (str): 
            From which day to get data. for example: '01-01-2001'
        end_date (str): 
            Till which day to get data. for example: '31-12-2022'
        interval (str): 


    Returns:
        _type_: _description_
    """

    data = yf.download(ticker,start_date, end_date, interval=interval)
    return data

def getTickerInfo(tickerSymbol):
    """_summary_

    Args:
        tickerSymbol (str):
            - Enter ticker symbol as an input.

    Returns:
        str:
            - Provides the detailed information about the Tocken.
    """
    tickerData = yf.Ticker(tickerSymbol)
    string_name = tickerData.info['description']
    return string_name

def getTickerLogo(tickerSymbol):
    """_summary_

    Args:
        tickerSymbol (str):
            - Enter ticker symbol as an input.

    Returns:
        str:
            - Provides the logo of the Tocken.
    """
    tickerData = yf.Ticker(tickerSymbol)
    string_logo = tickerData.info
    return string_logo

def exportFile(data, path, folder, tocken, edate):    
    
    if not os.path.exists(f"{path}/{folder}"):
        print(f"""
            Status: Creating New Directory || path: {path}/{folder}
            """)
        os.mkdir(f"{path}/{folder}")
        os.mkdir(f"{path}/{folder}/{tocken}")
        data.to_feather(f"{path}/{folder}/{tocken}/{edate}.ftr")
    
    else:
        if not os.path.exists(f"{path}/{folder}/{tocken}"):
            os.mkdir(f"{path}/{folder}/{tocken}")
            data.to_feather(f"{path}/{folder}/{tocken}/{edate}.ftr")
        else:
            data.to_feather(f"{path}/{folder}/{tocken}/{edate}.ftr")
    print(f"Ticker: {tocken} || Status: Done...", end = '\r')
    del data

def exportAllHistoricalData(level='1d', start_date=None, end_date=None, export_path = None):
    
    # parameter condition checks
    assert isinstance(level,str), "Parameter: 'level' must be string. Choose either of following: ['1m','15m','1hr','daily', 'weekly', 'monthly']"
    #assert isinstance(start_date,dt.date), "Parameter:'start_date' must be either date object or string date (For Example: dd-mm-yyyy)."
    #assert isinstance(end_date,dt.date), "Parameter:'end_date' must be either date object or string date (Format: dd-mm-yyyy)."
    #assert isinstance(export_path,str), "Enter Valid Path !!!"
    
    print(f"""
            ==================================================================
                            Processing Summary - {level}
            ==================================================================
            """)
    print("""Status: Tickers Information Loading...............................""", end = '\r')
    #get all the tickers
    tickers = getTickers()
    print(f"""
        Status: Tickers Loaded............................................
        Total Number of Tickers: {len(tickers)}
        """, end = '\r')
    
    if start_date == None:
        start_date = dt.datetime(2010,1,1)
    else:
        start_date = dt.datetime.strptime(start_date,'%d-%m-Y').date()
        
    if end_date == None:
        end_date = dt.datetime.now()
    else:
        end_date = dt.datetime.strptime(end_date,'%d-%m-Y').date()
    
    # fetching time delta and folder names
    timedelta = getTimeDelta()
    foldernames = getFolderNames()
    
    # Lower Frequencies
    if (level == '1d' or level == '5d' or level == '1wk' or level == '1mo' or level == '3mo' ):
        
        for ticker in tqdm(tickers, desc='process tickers Data'):
            
            print(f"""
            Ticker: {ticker} || Status: Processing Ticker data...""", end = '\r')
            
            res = getTickerData(ticker,
                                start_date = start_date,
                                end_date = end_date,
                                interval = level)
            res = res.reset_index().rename({'index':'Date'},axis=1)

            print(f"""Ticker: {ticker} || Status: Exporting Ticker data... || shape: {res.shape}""", end = '\r')

            if res is not None:
            
                exportFile(data=res, 
                           path = export_path, 
                           folder=foldernames.get(level),
                           tocken=ticker,
                           end_date=end_date
                           )
            else:
                del res
                continue
        
    # Moderate frequencies
    if (level == '1m' or level == '2m' or level == '5m' or level == '30m' or 
        level == '60m' or level == '1hr' or level == '90m'):
        
        for ticker in tqdm(tickers, desc='process tickers Data'):
            
            print(f"Ticker: {ticker} || Status: Processing Ticker data...", end = '\r')
            
            sd = dt.datetime.now().date()
            delta = timedelta.get(level)
            ed = ed = sd+ rtd(days = delta)
            
            res = getTickerData(ticker,
                                start_date = ed,
                                end_date = sd,
                                interval=level)
            res = res.reset_index().rename({'index':'Date'},axis=1)
            
            if res is not None:
                #print(f"{ticker} has data...",end = '\r')
                exportFile(data = res, 
                           path = export_path, 
                           folder = foldernames.get(level), 
                           tocken = ticker, 
                           edate = sd)
            else:
                del res
                print(f"{ticker} has skipped...")
                continue

