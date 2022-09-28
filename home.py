import streamlit as st

def app():
    st.title("Currency Forecast") #title of the web app
    st.markdown("""
                This app predicts the price of crypto currency
                
                **Credits**
                - App built by [Jay Savani](https://github.com/jaysavani1)
                - Built in `Python` using `streamlit`,`yfinance`, `cufflinks`, `pandas` and `datetime`
                
                
                """) #about section
    st.write("---")
    st.text_input("", "Search...") #search box