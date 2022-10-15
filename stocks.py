from imports import *
from forecast_model import forecast_bse

def app():
    #title of the web app
    
    b = BSE()
    b = BSE(update_codes = True) # to execute "updateScripCodes" on instantiation
    c_df = b.getScripCodes() # retrive key,company pair dictionary
    p = dict(zip(c_df.values(), c_df.keys())) # reverse the dictionary
    stock_list = list(p.keys()) 
    period_list = ['12M','6M','3M','1M']
    
    # Streamlit page
    st.markdown("<h1 style='text-align: center;'>Stocks Predictions</h1>", unsafe_allow_html=True) # Main heading
    st.sidebar.subheader("Parameters") # Sidebar
    stockSymbol = st.sidebar.selectbox('Company', stock_list) # Select company
    timePeriod = st.sidebar.selectbox('Time Duration', period_list) # Select time duration
    future_days = st.sidebar.slider('Days', 7,90,14)
    
    st.subheader(f"{stockSymbol} - {timePeriod}")
    
    q = b.getPeriodTrend(p[stockSymbol],timePeriod)
    q_df = pd.DataFrame(q)
    q_df = q_df.rename({'date':'Date', 'value':'Close','vol':'Volume'},axis=1)
    q_df['Date'] = pd.to_datetime(q_df['Date']).dt.date
    print(q_df)
    st.write(q_df)
    
    col1, col2 = st.columns(2)

    col1.markdown("<h3 style='text-align: center;'>Price</h3>", unsafe_allow_html=True)
    col1.line_chart(data = q_df,
                  x = 'Date',
                  y = 'Close')
    col2.markdown("<h3 style='text-align: center;'>Volume</h3>", unsafe_allow_html=True)
    col2.bar_chart(data = q_df,
                    x = 'Date',
                    y = 'Volume')
    
    if st.sidebar.button('Predict'):
        df_forecast = forecast_bse(q_df, days = future_days)
        st.area_chart(df_forecast,  
                      x = 'Date',
                      y = 'Close',
                      #width = 5.5,
                      use_container_width=True)
    
    
    #start_date = st.sidebar.date_input("Start date", dt.date(2015, 1, 1))
    #end_date = st.sidebar.date_input("End date", dt.date.today())
    
    
    
    # # Fetch data
    # df = ut.getTickerData("AAPL",start_date=start_date,end_date=end_date,interval='1d')
    # st.write(df)
    # fig = vis.plotTockenData(df=df, name=tickerSymbol,title="Chart")
    # fig = fig.iplot(asFigure=True)
    
    #st.plotly_chart(fig)
    
def checkBSE(stock, duration):
    pass
    # plt.plot(q_df['date'], q_df['value'])
    # plt.title(stock)
    # plt.show()