import streamlit as st

def app():
    st.title("Currency Forecast") #title of the web app
    st.markdown("""
                This app predicts the price of crypto currency
                
                **Credits**
                - App built by [Jay Savani](https://github.com/jaysavani1)
                - Built in `Python` using `streamlit`
                - Disclaimer: This app is created for demonstration purpose only.
                    Do not rely on predicted price. Do ypur own research before investing.
                """) #about section
    st.write("---")