import streamlit as st

class MultiLayout:
    
    def __init__(self) -> None:
        self.apps = []
    
    def add_category(self,title,fuction):
        """Add a new category. For Example: Cypto, Stocks, etc.

        Args:
            title (str): 
                title of the app. Appears in the dropdown in the sidebar.
            fuction (_type_):
                python function to render this app.
        """
        self.apps.append({
            "title":title,
            "function":fuction
        })
    
    def run(self):
        app = st.sidebar.radio(
            'Categories',
            self.apps,
            format_func = lambda app: app['title']
        )
        
        app['function']()