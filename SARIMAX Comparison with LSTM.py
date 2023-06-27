#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np # Load the data i
import warnings
warnings.filterwarnings('ignore')
import math
def tst(num=1000):
    arr=[]
    for i in range(0,num):
      arr.append(i*i+i)
    return arr


n_lookback=250
n_forecast=20
def create_sequence(dataset ):
     X = []
     Y = []
     for i in range(n_lookback, len(dataset) - n_forecast + 1):
        X.append(dataset[i - n_lookback: i])
        Y.append(dataset[i: i + n_forecast])

     X = np.array(X)
     Y = np.array(Y)

     return(X,Y)





dtmp=tst()


df=pd.DataFrame(data=dtmp,columns=['Value'])
df['Date']=range(0,len(dtmp))
df=df[['Date','Value']]
df=df.set_index('Date')




#df = yf.download(coin_,  start='2022-01-01', end=dt.now() , interval=interval_) #data = pd.read_csv('data.csv', index_col='Date', parse_dates=True)

# Set the frequency of the index to monthly #
#data = data.resample('M').asfreq()
data=df
data.dropna().astype (float)

# Plot the data
data.plot()
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Check the stationarity of the time series
result = adfuller(data['Value'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critical Values:')
    print(f'   {key}: {value}')

# Apply first-order differencing to make the time series stationary
diffs = data.diff().dropna()

# Plot the differenced data
diffs.plot()
plt.title('Differenced Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()
##-----------------------------------------------------------------------------------------
# Check the stationarity of the differenced time series
result = adfuller(diffs['Value'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critical Values:')
    print(f'   {key}: {value}')

# Define the SARIMAX model
model = SARIMAX(endog=diffs, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))

# Fit the SARIMAX model
results = model.fit()

# Print the model summary
print(results.summary())

# Make predictions for the next 100 periods
forecast = results.forecast(steps=n_forecast)

# Convert the differenced forecast back to the original scale
last_value = data.iloc[-1]['Value']
forecast = forecast.cumsum() + last_value

# Plot the predicted values
plt.plot(df['Value'])
plt.plot(forecast, color='red')
plt.title('Predicted Values')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

#-------------------------LSTM

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaler=scaler.fit(df)
dft=scaler.transform(df)

training_size = round(len(dft) * 0.7)
train_data = dft[:training_size]
test_data = dft[training_size:]

X,Y=create_sequence(train_data)
Xtest,Ytest=create_sequence(test_data)
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM, Bidirectional


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.4))
model.add(LSTM(units=50))
model.add(Dropout(0.4))
model.add(Dense(n_forecast))

model.compile(loss='mean_squared_error', metrics=['mean_absolute_error'], optimizer='adam')
model.summary()
history=model.fit(X, Y, epochs=12, batch_size=128, verbose=1, validation_data=(Xtest,Ytest))

plt.plot(history.history['loss'], 'r', label='Training loss')
plt.plot(history.history['val_loss'], 'g', label='Validation loss')
plt.title('Training VS Validation loss')
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


X_ = dft[- n_lookback:].reshape(1,-1)

Y_=model.predict(X_)

YP=scaler.inverse_transform(Y_)

YP=pd.DataFrame(data=YP.T,columns=["Value"])
YP['Date']=range(len(dtmp),len(dtmp)+n_forecast)
YP=YP.set_index('Date')
# Convert the differenced forecast back to the original scale

# Plot the predicted values
plt.plot(df['Value'])
plt.plot(YP, color='red')
plt.title('Predicted LSTM')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




