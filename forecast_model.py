#utilities
import utilities as ut
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from multiprocessing import Pool

# Model Forcasting requirements
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Additional packages       
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# btc = ut.getTickerData('XRP-USD', '2015-01-01', '2022-09-01', '1d')
# print


def for_cryp(data,train_dates,days, column: str, plot_lag_days = 3):
    #print(data)
    #Variables for training
    #cols = list(data)[1:7]
    cols = [column, 'Volume']
    print("training_cols: ",cols)

    #New dataframe with only training data - 5 columns
    df_for_training = data[cols].astype(float)
    print("df_for_training:", df_for_training)
    
    #df_for_plot = df_for_training.tail(90)
    #df_for_plot.plot.line()
    
    #LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)
    print("df_for_training_scaled:", df_for_training_scaled)
    
    #As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
    #In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training).
    
    #Empty lists to be populated using formatted training data
    trainX = []
    trainY = []
    
    n_future = 1   # Number of days we want to look into the future based on the past days.
    n_past = 14  # Number of past days we want to use to predict the future.
    
    for i in range(n_past, len(df_for_training_scaled) - n_future +1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)

    print('trainX shape == {}.'.format(trainX.shape))
    print('trainY shape == {}.'.format(trainY.shape))
    
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # fit the model
    history = model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

    # plt.plot(history.history['loss'], label='Training loss')
    # plt.plot(history.history['val_loss'], label='Validation loss')
    # plt.legend()
    # plt.show()
    
    #Remember that we can only predict one day in future as our model needs 5 variables
    #as inputs for prediction. We only have all 5 variables until the last day in our dataset.
    #n_past = 16
    n_days_for_prediction=days  #let us predict past 15 days
    #n_future = 15

    predict_period_dates = pd.date_range(list(train_dates)[-plot_lag_days], periods=n_days_for_prediction).tolist()
    #print(predict_period_dates)
    
    #Make prediction
    prediction = model.predict(trainX[-n_days_for_prediction:]) #shape = (n, 1) where n is the n_days_for_prediction

    #Perform inverse transformation to rescale back to original range
    #Since we used 5 variables for transform, the inverse expects same dimensions
    #Therefore, let us copy our values 5 times and discard them after inverse transform
    prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]
    
    # Convert timestamp to date
    forecast_dates = []
    for time_i in predict_period_dates:
        forecast_dates.append(time_i.date())
        
    df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), f'{column}':y_pred_future})
    df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

    # original = data[['Date', 'Close']]
    # original['Date']=pd.to_datetime(original['Date']).dt.date
    # original = original.loc[original['Date'] >= dt.date(2022,5,1)]

    # sns.plot(original['Date'], original['Close'])
    # sns.plot(df_forecast['Date'], df_forecast['Close'])
    # plt.show()
    df_forecast = df_forecast.set_index("Date")
    
    return df_forecast

def forecast_bse(data,days):
    #print(data)    
    
    print(data)
    data = data.reset_index()
    #data['Date'] = pd.to_datetime(data['Date']).dt.date
    train_dates = pd.to_datetime(data['Date'])
    #Variables for training
    cols = list(data)[2:4]
    print(cols)

    #New dataframe with only training data - 5 columns
    df_for_training = data[cols].astype(float)
    
    df_for_plot = df_for_training.tail(90)
    df_for_plot.plot.line()
    
    #LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)
    
    #As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
    #In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training).
    
    #Empty lists to be populated using formatted training data
    trainX = []
    trainY = []
    
    n_future = 1   # Number of days we want to look into the future based on the past days.
    n_past = 30  # Number of past days we want to use to predict the future.
    
    for i in range(n_past, len(df_for_training_scaled) - n_future +1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)

    print('trainX shape == {}.'.format(trainX.shape))
    print('trainY shape == {}.'.format(trainY.shape))
    
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # fit the model
    history = model.fit(trainX, trainY, epochs=25, batch_size=16, validation_split=0.1, verbose=1)

    # plt.plot(history.history['loss'], label='Training loss')
    # plt.plot(history.history['val_loss'], label='Validation loss')
    # plt.legend()
    # plt.show()
    
    #Remember that we can only predict one day in future as our model needs 5 variables
    #as inputs for prediction. We only have all 5 variables until the last day in our dataset.
    #n_past = 16
    n_days_for_prediction=days  #let us predict past 15 days
    #n_future = 15

    predict_period_dates = pd.date_range(list(train_dates)[-5], periods=n_days_for_prediction,freq=us_bd).tolist()
    #print(predict_period_dates)
    
    #Make prediction
    prediction = model.predict(trainX[-n_days_for_prediction:]) #shape = (n, 1) where n is the n_days_for_prediction

    #Perform inverse transformation to rescale back to original range
    #Since we used 5 variables for transform, the inverse expects same dimensions
    #Therefore, let us copy our values 5 times and discard them after inverse transform
    prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]
    
    # Convert timestamp to date
    forecast_dates = []
    for time_i in predict_period_dates:
        forecast_dates.append(time_i.date())
        
    df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Close':y_pred_future})
    df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

    # original = data[['Date', 'Close']]
    # original['Date']=pd.to_datetime(original['Date']).dt.date
    # original = original.loc[original['Date'] >= dt.date(2022,5,1)]

    # sns.plot(original['Date'], original['Close'])
    # sns.plot(df_forecast['Date'], df_forecast['Close'])
    # plt.show()
    return df_forecast
    
def forecast_crypto(data, days):
    #print(data)
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    t_dates = pd.to_datetime(data['Date'])
    with Pool(4) as p:
        df = p.starmap(for_cryp, [(data, t_dates, days, 'Open'),
                                (data, t_dates, days, 'High'),
                                (data, t_dates, days, 'Low'),
                                (data, t_dates, days, 'Close')])
    try:
        res_df = pd.concat(df, axis = 1).round(4)
        # res_df = res_df.reset_index()
        # res_df['Date'] = pd.to_datetime(res_df['Date']).dt.date
        if res_df.empty:
            raise ValueError("Prediction data merger failed!!")
        return res_df
    except Exception as e:
        print(e)