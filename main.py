import os
import pandas as pd
import numpy as np
import utilities as ut
import datetime as dt

def main():
    
    exp_path = 'E:/Workplace/Coding/Projects/Cryps/daily_data'
    ut.exportAllHistoricalData(export_path=exp_path)
    
    # data = getTickerData('DENT-USD',
    #         start_date = dt.datetime(2010,1,1),
    #         end_date = dt.datetime.today(),
    #         interval ='1d')
    
    # print(data)
if __name__ == '__main__':
    main()