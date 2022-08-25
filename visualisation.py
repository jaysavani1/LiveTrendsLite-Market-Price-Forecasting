from unicodedata import name
import plotly.graph_objects as go
import indicators as ind
import utilities as ut

def plotTockenData(df,sma = False,sma_period = 21,ema = False,ema_period = 21, bbbands = False,title=None,xlabel = 'Date', ylabel='Price'):
    """_summary_

    Args:
        df (_type_): _description_
        sma (bool, optional): _description_. Defaults to False.
        sma_period (int, optional): _description_. Defaults to 21.
        ema (bool, optional): _description_. Defaults to False.
        ema_period (int, optional): _description_. Defaults to 21.
        bbbands (bool, optional): _description_. Defaults to False.
        title (_type_, optional): _description_. Defaults to None.
        xlabel (str, optional): _description_. Defaults to 'Date'.
        ylabel (str, optional): _description_. Defaults to 'Price'.
    """
    
    closing_price = df['Close']
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name = f'{title.upper()} candles'))
    if sma:
        plainsma = ind.getSMA(price = closing_price, period=sma_period)
        fig.add_trace(go.Line(x=df['Date'],y = plainsma,
                    name = f'SMA - {sma_period}'))
        
    if ema:
        plainsma = ind.getEMA(price = closing_price, period=ema_period)
        fig.add_trace(go.Line(x=df['Date'],y = plainsma,
                    name = f'EMA - {ema_period}'))

    if bbbands:
        bbupper,bbmiddle, bblower = ind.getBollingerBands(closing_price)
        fig.add_trace(go.Line(x=df['Date'],y = bbupper,
                    name = 'Bollinger Up'))
        fig.add_trace(go.Line(x=df['Date'],y = bblower,
                    name='Bollinger Low'))
        fig.add_trace(go.Line(x=df['Date'],y = bbmiddle,
                    name='Bollinger Middle'))


    fig.update_layout(title ={
                        'text': f"{title.upper()}",
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                    )
    fig.show()
    
    
    
    