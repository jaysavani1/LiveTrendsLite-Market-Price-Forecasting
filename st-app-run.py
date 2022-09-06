import crypto, home, stocks
import streamlit as st
from multilayout import MultiLayout

def main():
    
    app = MultiLayout()
    #page layout to full width
    st.set_page_config(layout='wide')
    
    app.add_category("Home",home.app)
    app.add_category("Cryptocurrency",crypto.app)
    app.add_category("Stocks",stocks.app)
    
    app.run()
    
if __name__ == '__main__':
    main()