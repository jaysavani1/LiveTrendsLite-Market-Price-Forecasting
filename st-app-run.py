import os
import pandas as pd
import numpy as np
import streamlit as st

import utilities as ut
import datetime as dt
import visualisation as vis

def main():
    
    st.set_page_config(layout='wide') #page layout to full width
    st.title("Crypto Currency Predictions") #title of the web app
    st.markdown("""
                This app predicts the price of crypto currency
                
                **Credits**
                - App built by [Jay Savani](https://github.com/jaysavani1)
                - Built in `Python` using `streamlit`,`yfinance`, `cufflinks`, `pandas` and `datetime`
                
                
                """) #about section
    st.write("---")
    st.text_input("", "Search...") #search box
    st.sidebar.subheader("Parameters")
    start_date = st.sidebar.date_input("Start date", dt.date(2015, 1, 1))
    end_date = st.sidebar.date_input("End date", dt.date.today())
    ticker_list = ut.getTickers()
    tickerSymbol = st.sidebar.selectbox('Coin', ticker_list) # Select ticker symbol
    
    
    # Fetch data
    df = ut.getTickerData(tickerSymbol,start_date=start_date,end_date=end_date,interval='1d')
    fig = vis.plotTockenData(df=df, name=tickerSymbol,title="Chart")
    fig = fig.iplot(asFigure=True)
    
    st.plotly_chart(fig)


if __name__ == '__main__':
    main()