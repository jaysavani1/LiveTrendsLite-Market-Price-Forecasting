import streamlit as st
import utilities as ut
import datetime as dt
import visualisation as vis
from forecast_model import forecast_crypto

def app():
    
    #title of the web app
    st.title("Crypto Currency Predictions")
    st.sidebar.caption("Parameters")
    col1, col2, col3 = st.columns(3)
    select = ['--Select--']
    ticker_list = select + ut.getTickers()
    tickerSymbol = col1.selectbox('Coin', ticker_list) # Select ticker symbol
    
    #@st.experimental_memo
    # Fetch data
    if tickerSymbol != select[0]:
        start_date = col2.date_input("Start date", dt.date.today() - dt.timedelta(days=183))
        end_date = col3.date_input("End date", dt.date.today())
        future_days = st.sidebar.slider('Days', 7,90,7)
        past_num_days = (start_date - end_date).days
        df = ut.getTickerData(tickerSymbol,start_date=start_date,end_date=end_date,interval='1d')
        df_today = ut.getTickerData(tickerSymbol,start_date=start_date, end_date=dt.datetime.now() ,interval='1d')
        #st.write(df_today)
        st.header("Todays' Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Open", df_today['Open'][-1].round(2),(df_today['Open'][-1] - df_today['Open'][-2]).round(4))
        col2.metric("High", df_today['High'][-1].round(2),(df_today['High'][-1] - df_today['High'][-2]).round(4))
        col3.metric("Low", df_today['Low'][-1].round(2),(df_today['Low'][-1] - df_today['Low'][-2]).round(4))
        col4.metric("Close", df_today['Close'][-1].round(2),(df_today['Close'][-1] - df_today['Close'][-2]).round(4))

        fig = vis.plotTockenData(df=df, name=tickerSymbol,title="Chart", add_rsi=True, add_vol = True,add_bbands=True)
        fig = fig.iplot(asFigure=True)
        
        if tickerSymbol!='--Select--':
            sb_col1,sb_col2 = st.sidebar.columns(2)
            sb_col1.caption("Prediction")
            sb_col2.caption("Clear")
            
            st.header("Chart")
            st.plotly_chart(fig,use_container_width=True)

            if sb_col1.button('Predict'):
                with st.spinner('Wait for it...'):
                    res_df = forecast_crypto(data = df, days = future_days)
                    #st.dataframe(res_df)
                    st.header("Predicted Price")
                    # fig2 = vis.plotTockenData(df=res_df,
                    #                         kind="candle", keys=["High","Low"]
                    #                         , name=f"Predicted-{tickerSymbol}",title="Chart")
                    # fig2 = fig2.iplot(asFigure=True)
                    # st.plotly_chart(fig2,use_container_width=True)
                    st.balloons()
                    fig2 = vis.predictedPlot(data = res_df)
                    st.plotly_chart(fig2,use_container_width=True)
                    st.caption(f"This prices are predicted based on last {past_num_days} days!!!")
            
            if sb_col2.button("Clear"):
                st.runtime.legacy_caching.clear_cache()
                #st.experimental_memo.clear()