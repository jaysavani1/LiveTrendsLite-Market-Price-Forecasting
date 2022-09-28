import streamlit as st
import utilities as ut
import datetime as dt
import visualisation as vis

def app():
    #title of the web app
    st.title("Crypto Currency Predictions")
    st.sidebar.caption("Parameters")
    col1, col2, col3 = st.columns(3)
    ticker_list = ut.getTickers()
    tickerSymbol = col1.selectbox('Coin', ticker_list) # Select ticker symbol
    start_date = col2.date_input("Start date", dt.date.today() - dt.timedelta(days=365))
    end_date = col3.date_input("End date", dt.date.today())
    st.sidebar.caption("Prediction")
    future_days = st.sidebar.slider('Days', 1,7,1)
    
    # Fetch data    
    df = ut.getTickerData(tickerSymbol,start_date=start_date,end_date=end_date,interval='1d')
    fig = vis.plotTockenData(df=df, name=tickerSymbol,title="Chart")
    fig = fig.iplot(asFigure=True)

    st.plotly_chart(fig,use_container_width=True)
    
    if st.sidebar.button('Predict'):
        pass