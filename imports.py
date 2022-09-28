# Standard module packages
import os
#from symtable import Symbol
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta as rtd
#from unicodedata import name
from tqdm import tqdm
import matplotlib.pyplot as plt
#from re import L
#from xml.sax.saxutils import prepare_input_source

# Streamlit UI + Plots
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default='browser'

# Custom made modules
import indicators as ind
import utilities as ut
import cufflinks as cf
import utilities as ut
import visualisation as vis

# Dataset APIs => Stocks + Crypto
from yahooquery import Screener
import yfinance as yf
from bsedata.bse import BSE

