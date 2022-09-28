from imports import *

def main():
    
    # fetch all the levels
    lvls = list(ut.getTimeDelta().keys())
    exp_path = 'Cryps/'
    
    for lvl in lvls:
        ut.exportAllHistoricalData(level = lvl,export_path=exp_path)
        
def test():
    # print("hello")
    # s = ut.getTickerLogo('BTC-USD')
    # print(s)
    print(dt.date.today() - dt.timedelta(days=90))
           
if __name__ == '__main__':
    test()