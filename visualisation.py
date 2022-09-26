from imports import *
cf.go_offline()

def oldplotTockenData(df,date_col,sma = False,sma_period = 21,ema = False,ema_period = 21, bbbands = False,title=None,xlabel = 'Date', ylabel='Price'):
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

    fig.add_trace(go.Candlestick(x=df[date_col],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name = f'{title.upper()} candles',
                    increasing_line_color= 'green',
                    decreasing_line_color = 'red',
                    )
                  )
    if sma:
        plainsma = ind.getSMA(price = closing_price, period=sma_period)
        fig.add_trace(go.Line(x=df[date_col],y = plainsma,
                    name = f'SMA - {sma_period}'))
        
    if ema:
        plainsma = ind.getEMA(price = closing_price, period=ema_period)
        fig.add_trace(go.Line(x=df[date_col],y = plainsma,
                    name = f'EMA - {ema_period}'))

    if bbbands:
        bbupper,bbmiddle, bblower = ind.getBollingerBands(closing_price)
        fig.add_trace(go.Line(x=df[date_col],y = bbupper,
                    name = 'Bollinger Up'))
        fig.add_trace(go.Line(x=df[date_col],y = bblower,
                    name='Bollinger Low'))
        fig.add_trace(go.Line(x=df[date_col],y = bbmiddle,
                    name='Bollinger Middle'))


    fig.update_layout(title ={
                        'text': f"{title.upper()}",
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                    xaxis_rangeslider_visible=False,
                    height=600,
                    
                    )
    fig.show()
    
    
def plotTockenData(df, title,name,legend='top',theme='pearl',up_color='green',down_color='red',dimensions =(880,650)):
    
    qf=cf.QuantFig(
            df = df,
            title=title,
            legend=legend,
            name=name,
            theme=theme,
            up_color=up_color,
            down_color=down_color,
            dimensions = dimensions
        )
    qf.add_volume()
    qf.add_rsi(name = 'RSI',showbands=True, legendgroup = True)            
    qf.add_bollinger_bands(legendgroup = True)
    qf.add_sma([10,22,44,100,200],legendgroup = True)
    qf.add_ema([10,22,44,100,200],legendgroup = True)
    
    return qf
    