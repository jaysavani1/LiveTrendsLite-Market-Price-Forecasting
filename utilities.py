import os
import datetime as dt
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


def exportAllHistoricalData(level='daily', start_date=None, end_date=None, export_path = None):
    
    # parameter condition checks
    assert isinstance(level,str), "Parameter: 'level' must be string. Choose either of following: ['1m','15m','1hr','daily', 'weekly', 'monthly']"
    #assert isinstance(start_date,dt.date), "Parameter:'start_date' must be either date object or string date (For Example: dd-mm-yyyy)."
    #assert isinstance(end_date,dt.date), "Parameter:'end_date' must be either date object or string date (Format: dd-mm-yyyy)."
    assert isinstance(export_path,str), "Enter Valid Path !!!"
    
    print("""
            ==================================================================
                                        Processing Summary
            ==================================================================
            """)
    print("""Status: Tickers Information Loading...............................""")
    #get all the tickers
    tickers = getTickers()
    print(f"""
        Status: Tickers Loaded............................................
        Total Number of Tickers: {len(tickers)}
        Ticker: {list(tickers)}
        """)
    
    if start_date == None:
        start_date = dt.datetime(2010,1,1)
    else:
        start_date = dt.datetime.strptime(start_date,'%d-%m-Y').date()
        
    if end_date == None:
        end_date = dt.datetime.now()
    else:
        end_date = dt.datetime.strptime(end_date,'%d-%m-Y').date()
    
    if level == 'daily':
        
        delta = rtd(years=1)
        
        for ticker in tqdm(tickers, desc='process tickers Data'):
            
            print(f"""
            Ticker: {ticker} || Status: Processing Ticker data...""")
            
            #empty dataframe
            res = pd.DataFrame(columns=['Open','High','Low','Close','Adj Close','Volume'])
            
            sdate = start_date
            edate = end_date
            
            while sdate < edate:
                ndate = sdate + delta
                #print(startdate,nextdate)
                
                #get the 15 min weekly time frame data
                data = getTickerData(ticker,
                                    start_date = sdate,
                                    end_date = ndate,
                                    interval ='1d')
                res = pd.concat([res,data])
                
                sdate += delta

            print(f"""Ticker: {ticker} || Status: Exporting Ticker data... || shape: {res.shape}""")
            
            if res is not None:
            
                if not os.path.exists(f"{export_path}/{ticker}"):
                    print(f"""
                        Status: Creating New Directory || path: {export_path}/{ticker}
                        """, end='\r')
                    os.mkdir(f"{export_path}/{ticker}")
                    res.to_csv(f"{export_path}/{ticker}/{end_date.date()}.csv")
                
                else:
                    res.to_csv(f"{export_path}/{ticker}/{end_date.date()}.csv")
                print(f"Ticker: {ticker} || Status: Done...")
            
                del res
                del sdate
                del edate
                del ndate
            else:
                del res
                del sdate
                del edate
                del ndate
                continue
        
    
    