import streamlit as st

def app():
    st.title("Currency Forecast") #title of the web app
    st.markdown("""
                This app predicts the price of crypto currency and stock market. Advanced version is coming soon !!!
                
                **Credits**
                - App built by [Jay Savani](https://github.com/jaysavani1)
                - Built in: `Python` using `streamlit`, 'keras', 'pandas'
                - Prediction model: LSTMs with 3 Layer Network
                - NO FINANCIAL ADVISE. THIS APP IS CREATED FOR DEMONSTRATION OF DEEP LEARNING TECHNIQUES ONLY.
                        DO NOT RELY ON PREDICTED PRICE.
                """) #about section
    st.markdown("""- Disclaimer: Before using this site, please make sure that you note the following important information: 
                        Do your Own Research Our content is intended to be used and must be used for 
                        informational purposes only. 
                        It is very important to do your own analysis before making any investment 
                        based on your own personal circumstances. 
                        You should take independent financial advice from a professional in connection with, 
                        or independently research and verify, 
                        any information that you find on our Website and wish to rely upon, 
                        whether for the purpose of making an investment decision or otherwise., """)
    st.write("---")