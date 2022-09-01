from xml.sax.saxutils import prepare_input_source
import pandas as pd
import numpy as np

def getSMA(price:pd.Series, period = 21):
    """_summary_

    Args:
        closing_price (pd.Series):
            Enter the Closing price from the dataset 
        period (int, optional): _description_. Defaults to 21.
            average of total number of previous candles

    Returns:
        _type_: float
            simple moving average of price
    """
    if price is not None:
        return price.rolling(period).mean()
    else:
        assert "Invalid closing price !!!"
        
def getEMA(price:pd.Series,period):
    """_summary_

    Args:
        closing_price (pd.Series):
            Enter the Closing price from the dataset 
        period (int, optional): _description_. Defaults to 21.
            average of total number of previous candles

    Returns:
        _type_: float
            exponantial moving average of price
    """
    return price.ewm(span = period).mean()
    

def getBollingerBands(price:pd.Series, period = 21):
    """_summary_

    Args:
        price (pd.Series):
            Enter prices of tocken. For Example: df['close'] or df['high'] or df['low'], etc. 
        
        period (int, optional): _description_. Defaults to 21.
            BB bounds of total number of previous candles
    Returns:
        _type_: float
            returns upperbound,lowerbound
    """
    sma = getSMA(price=price, period=period)
    std = price.rolling(period).std()
    bbupperbound = sma + (std * 2)
    bbmiddlebound = sma
    bblowerbound = sma - (std * 2)
    
    return bbupperbound,bbmiddlebound,bblowerbound

def getCMA(price:pd.Series,period=1):
    """_summary_

    Args:
        closing_price (pd.Series):
            Enter the Closing price from the dataset 
        period (int, optional): _description_. Defaults to 1.
            average of total number of previous candles

    Returns:
        _type_: float
            cumilative moving average of price
    """
    return price.expanding(min_periods = period).mean()