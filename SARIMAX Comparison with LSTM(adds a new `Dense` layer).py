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





# In[4]:


Open In Colab


     

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np # Load the data i


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
model.add(Dense(64, activation='relu'))
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



     

ADF Statistic: -1.0343625929540556
p-value: 0.7405259806548131
Critical Values:
   1%: -3.4370062675076807
Critical Values:
   5%: -2.8644787205542492
Critical Values:
   10%: -2.568334722615888

ADF Statistic: -7.56194937618537
p-value: 2.9938941433740185e-11
Critical Values:
   1%: -3.4370266558635914
Critical Values:
   5%: -2.864487711945291
Critical Values:
   10%: -2.5683395116993872
usr/local/lib/python3.10/dist-packages/statsmodels/tsa/base/tsa_model.py:471:(ValueWarning:, An, unsupported, index, was, provided, and, will, be, ignored, when, e.g., forecasting.)
  self._init_dates(dates, freq)
usr/local/lib/python3.10/dist-packages/statsmodels/tsa/base/tsa_model.py:471:(ValueWarning:, An, unsupported, index, was, provided, and, will, be, ignored, when, e.g., forecasting.)
  self._init_dates(dates, freq)
usr/local/lib/python3.10/dist-packages/statsmodels/tsa/statespace/sarimax.py:966:(UserWarning:, Non-stationary, starting, autoregressive, parameters, found., Using, zeros, as, starting, parameters.)
  warn('Non-stationary starting autoregressive parameters'
/usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:604: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
/usr/local/lib/python3.10/dist-packages/statsmodels/tsa/base/tsa_model.py:834: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  Value   No. Observations:                  999
Model:               SARIMAX(1, 0, 0)   Log Likelihood               -2116.533
Date:                Wed, 21 Jun 2023   AIC                           4237.067
Time:                        10:04:25   BIC                           4246.880
Sample:                             0   HQIC                          4240.797
                                - 999                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          1.0000   2.01e-06   4.97e+05      0.000       1.000       1.000
sigma2         3.9997   5.07e-13   7.89e+12      0.000       4.000       4.000
===================================================================================
Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):          41278976.44
Prob(Q):                              0.99   Prob(JB):                         0.00
Heteroskedasticity (H):               1.00   Skew:                           -31.56
Prob(H) (two-sided):                  0.97   Kurtosis:                       996.83
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
[2] Covariance matrix is singular or near-singular, with condition number    inf. Standard errors may be unstable.

Model: "sequential_7"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm_17 (LSTM)              (None, 250, 50)           10400     
                                                                 
 dropout_17 (Dropout)        (None, 250, 50)           0         
                                                                 
 lstm_18 (LSTM)              (None, 50)                20200     
                                                                 
 dropout_18 (Dropout)        (None, 50)                0         
                                                                 
 dense_11 (Dense)            (None, 64)                3264      
                                                                 
 dense_12 (Dense)            (None, 20)                1300      
                                                                 
=================================================================
Total params: 35,164
Trainable params: 35,164
Non-trainable params: 0
_________________________________________________________________
Epoch 1/12
4/4 [==============================] - 7s 755ms/step - loss: 0.0703 - mean_absolute_error: 0.2366 - val_loss: 0.8027 - val_mean_absolute_error: 0.8933
Epoch 2/12
4/4 [==============================] - 3s 717ms/step - loss: 0.0567 - mean_absolute_error: 0.2066 - val_loss: 0.6338 - val_mean_absolute_error: 0.7845
Epoch 3/12
4/4 [==============================] - 2s 437ms/step - loss: 0.0388 - mean_absolute_error: 0.1584 - val_loss: 0.3808 - val_mean_absolute_error: 0.5683
Epoch 4/12
4/4 [==============================] - 2s 439ms/step - loss: 0.0255 - mean_absolute_error: 0.1240 - val_loss: 0.2526 - val_mean_absolute_error: 0.4574
Epoch 5/12
4/4 [==============================] - 2s 415ms/step - loss: 0.0160 - mean_absolute_error: 0.0972 - val_loss: 0.2600 - val_mean_absolute_error: 0.4938
Epoch 6/12
4/4 [==============================] - 2s 419ms/step - loss: 0.0111 - mean_absolute_error: 0.0798 - val_loss: 0.1843 - val_mean_absolute_error: 0.4157
Epoch 7/12
2/4 [==============>...............] - ETA: 0s - loss: 0.0085 - mean_absolute_error: 0.0701

1/1 [==============================] - 2s 2s/step


# In[5]:


{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNSQW8RrpVUk3Zp3yJ+xt3e",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/mohammadalaeifar/SARIMAX/blob/main/SARIMAX%20Comparison%20with%20LSTM(adds%20a%20new%20%60Dense%60%20layer).ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "su4wNtV5elZi"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "Yp_bfLFNyyo-",
        "outputId": "cbd27f65-6d04-497a-a67a-f668773b59ec"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "ADF Statistic: -1.0343625929540556\n",
            "p-value: 0.7405259806548131\n",
            "Critical Values:\n",
            "   1%: -3.4370062675076807\n",
            "Critical Values:\n",
            "   5%: -2.8644787205542492\n",
            "Critical Values:\n",
            "   10%: -2.568334722615888\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkQAAAHHCAYAAABeLEexAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABquklEQVR4nO3dd3gU5f7+8fem94QASQgECL0FCKCIUgUJCCqCIkVBQTl6AlIsiB0LICpIU4/nKHg8dAVUFJUqIBGQEHoXCC0JLQkhpO7z+4Mf+3UFMYEkm3K/rmuvi515ZuYzk03mZp5nZi3GGIOIiIhIGebk6AJEREREHE2BSERERMo8BSIREREp8xSIREREpMxTIBIREZEyT4FIREREyjwFIhERESnzFIhERESkzFMgEhERkTJPgUikELz++utYLBa7aTk5OTz//POEhYXh5OREjx49AEhLS+Pxxx8nJCQEi8XCiBEjir7gYmTWrFlYLBaOHDlSIOtr37497du3L5B1lURlff9F8kqBSORvXDlBX3l5eHgQGhpKVFQUU6dO5cKFC3laz2effca7777LAw88wOeff87IkSMBGDduHLNmzeKpp57iiy++4JFHHinM3Snxjhw5YvfzuN6roEJVQduxYwcPPPAA1apVw8PDg8qVK3PXXXcxbdo0R5dWIKpXr277GTg5OREQEEBERARDhgxh48aNN7XucePGsWTJkoIpVOQPLPouM5HrmzVrFo899hhvvPEG4eHhZGdnk5CQwJo1a1i+fDlVq1blm2++oXHjxrZlcnJyyMnJwcPDwzatT58+rF+/nuPHj9ut/7bbbsPFxYX169cX2T4VZ1eO9+HDh6levfpV8y9evMjixYvtpr3//vscP36cyZMn202///77cXV1BcDNza3Qas6PDRs20KFDB6pWrcrAgQMJCQnh2LFj/Prrrxw6dIiDBw8W6PaysrKAot3/6tWrU65cOZ555hkALly4wJ49e1i4cCEJCQmMHDmSSZMm3dC6fXx8eOCBB5g1a1YBViwCLo4uQKSk6Nq1Ky1atLC9HzNmDKtWraJ79+7ce++97NmzB09PTwBcXFxwcbH/9UpKSiIgIOCq9SYlJdGgQYMCq9NqtZKVlWUXxkoTb29vHn74Ybtp8+bN4/z581dNL47efvtt/P392bx581Wfh6SkpALbTnp6Ol5eXg4LgpUrV77q5/HOO+/Qr18/Jk+eTO3atXnqqaccUpvItajLTOQm3HnnnbzyyiscPXqU//3vf7bpfxxDdKWLZ/Xq1ezatcvWlbBmzRosFguHDx/mu+++u6qbJzMzk9dee41atWrh7u5OWFgYzz//PJmZmXY1WCwWhg4dyuzZs2nYsCHu7u788MMPAJw4cYJBgwYRHByMu7s7DRs25LPPPrNb/kodCxYs4O2336ZKlSp4eHjQsWPHa16t2LhxI3fffTflypXD29ubxo0bM2XKFLs2e/fu5YEHHiAwMBAPDw9atGjBN998c9W6du3axZ133omnpydVqlThrbfewmq15v8HcR1/HkPzx/0dO3YslStXxtfXlwceeICUlBQyMzMZMWIEQUFB+Pj48Nhjj111zAH+97//0bx5czw9PQkMDKRPnz4cO3bsb+s5dOgQDRs2vGY4DgoKuqHttG/fnkaNGrFlyxbatm2Ll5cXL7744jX3H/L+2Vq+fDmtW7cmICAAHx8f6tata1vvjfD09OSLL74gMDCQt99+mz92ULz33nvcfvvtlC9fHk9PT5o3b86XX35pt7zFYuHixYt8/vnntt+XRx99FICjR4/yz3/+k7p16+Lp6Un58uV58MEHi223qRQ/ukIkcpMeeeQRXnzxRX766SeeeOKJq+ZXrFiRL774grfffpu0tDTGjx8PQP369fniiy8YOXIkVapUsXUvVKxYEavVyr333sv69esZMmQI9evXZ8eOHUyePJn9+/dfNYZi1apVLFiwgKFDh1KhQgWqV69OYmIit912my0wVaxYkWXLljF48GBSU1OvGrw9YcIEnJycePbZZ0lJSWHixIn079/fbszH8uXL6d69O5UqVWL48OGEhISwZ88eli5dyvDhw4HLIeeOO+6gcuXKvPDCC3h7e7NgwQJ69OjBV199xf333w9AQkICHTp0ICcnx9buk08+sV1lK2zjx4/H09OTF154gYMHDzJt2jRcXV1xcnLi/PnzvP766/z666/MmjWL8PBwXn31Vduyb7/9Nq+88gq9e/fm8ccf5/Tp00ybNo22bduydevWa4adK6pVq0ZMTAw7d+6kUaNG160xP9s5e/YsXbt2pU+fPjz88MMEBwdfc515/Wzt2rWL7t2707hxY9544w3c3d05ePAgv/zyS56P8bX4+Phw//338+mnn7J7924aNmwIwJQpU7j33nvp378/WVlZzJs3jwcffJClS5fSrVs3AL744gsef/xxbr31VoYMGQJAzZo1Adi8eTMbNmygT58+VKlShSNHjvDRRx/Rvn17du/ejZeX103VLWWAEZHrmjlzpgHM5s2b/7KNv7+/iYyMtL1/7bXXzJ9/vdq1a2caNmx41bLVqlUz3bp1s5v2xRdfGCcnJ7Nu3Tq76R9//LEBzC+//GKbBhgnJyeza9cuu7aDBw82lSpVMmfOnLGb3qdPH+Pv72/S09ONMcasXr3aAKZ+/fomMzPT1m7KlCkGMDt27DDGGJOTk2PCw8NNtWrVzPnz5+3WabVabf/u2LGjiYiIMBkZGXbzb7/9dlO7dm3btBEjRhjAbNy40TYtKSnJ+Pv7G8AcPnz4qmP1V7p162aqVat2zXnt2rUz7dq1s72/sr+NGjUyWVlZtul9+/Y1FovFdO3a1W75Vq1a2a37yJEjxtnZ2bz99tt27Xbs2GFcXFyumv5nP/30k3F2djbOzs6mVatW5vnnnzc//vijXS353U67du0MYD7++OO/3f+8frYmT55sAHP69Onr7s+1XOsz/UdX1v3111/bpl35PF6RlZVlGjVqZO6880676d7e3mbgwIFXrfPPyxtjTExMjAHMf//733zugZRF6jITKQA+Pj55vtssLxYuXEj9+vWpV68eZ86csb3uvPNOAFavXm3Xvl27dnbjkIwxfPXVV9xzzz0YY+zWERUVRUpKCrGxsXbreOyxx+zGm7Rp0waA33//HYCtW7dy+PBhRowYcdUVkCvdg+fOnWPVqlX07t2bCxcu2LZ59uxZoqKiOHDgACdOnADg+++/57bbbuPWW2+1radixYr079//Zg5dng0YMMA24BqgZcuWGGMYNGiQXbuWLVty7NgxcnJyAFi0aBFWq5XevXvbHdeQkBBq16591c/mz+666y5iYmK499572bZtGxMnTiQqKorKlSvbdSvmdzvu7u489thjf7vfef1sXfkZf/311wXejenj4wNg9zvzxyuD58+fJyUlhTZt2lz1Of0rf1w+Ozubs2fPUqtWLQICAvK8Dinb1GUmUgDS0tKuOf7jRh04cIA9e/ZQsWLFa87/8+Db8PBwu/enT58mOTmZTz75hE8++SRP66hatard+3LlygGXT05weewLcN1unoMHD2KM4ZVXXuGVV175y+1WrlyZo0eP0rJly6vm161b9y/XX5D+vL/+/v4AhIWFXTXdarWSkpJC+fLlOXDgAMYYateufc31/jFk/ZVbbrmFRYsWkZWVxbZt21i8eDGTJ0/mgQceIC4ujgYNGuR7O5UrV87TAOq8frYeeugh/vOf//D444/zwgsv0LFjR3r27MkDDzyAk9PN/V86LS0NAF9fX9u0pUuX8tZbbxEXF2c3lunPz/P6K5cuXWL8+PHMnDmTEydO2I1PSklJual6pWxQIBK5ScePHyclJYVatWoV2DqtVisRERF/eWvyn0/afx53c+V/9A8//DADBw685jr++JgAAGdn52u2M/l4MseV7T777LNERUVds01BHqeb8Vf7+3fHwWq1YrFYWLZs2TXbXrn6kRdubm7ccsst3HLLLdSpU4fHHnuMhQsX8tprr+V7O3kde5XXz5anpydr165l9erVfPfdd/zwww/Mnz+fO++8k59++ukvj1Ne7Ny5E/i/z8K6deu49957adu2LR9++CGVKlXC1dWVmTNnMmfOnDytc9iwYcycOZMRI0bQqlUr/P39sVgs9OnTp8CvcEnppEAkcpO++OILgL8MADeiZs2abNu2jY4dO+b5f8h/VLFiRXx9fcnNzaVTp04FVhNcPpn91Tpr1KgBXL568XfbrVatGgcOHLhq+r59+26y0sJVs2ZNjDGEh4dTp06dAlvvlUc6nDp1qlC3k5/PlpOTEx07dqRjx45MmjSJcePG8dJLL7F69eob/lylpaWxePFiwsLCqF+/PgBfffUVHh4e/Pjjj7i7u9vazpw586rl/6rmL7/8koEDB/L+++/bpmVkZJCcnHxDdUrZozFEIjdh1apVvPnmm4SHhxfo2JfevXtz4sQJ/v3vf18179KlS1y8ePG6yzs7O9OrVy+++uor2//G/+j06dP5rqlZs2aEh4fzwQcfXHWSuXL1JCgoiPbt2/Ovf/3LdmL/q+3efffd/Prrr2zatMlu/uzZs/NdW1Hq2bMnzs7OjB079qqrZ8YYzp49e93lV69efc2rbt9//z3wf12GN7udv5LXz9a5c+eumt+0aVOAaz6GIC8uXbrEI488wrlz53jppZds4cbZ2RmLxUJubq6t7ZEjR675RGpvb+9rhhxnZ+erjtO0adPs1ilyPbpCJJJHy5YtY+/eveTk5JCYmMiqVatYvnw51apV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kUyZMoUqVarwn//8h6iowgktFoslz91Wjta7d2+GDx/OnDlz+O9//8tTTz2FxWLhl19+4b777uPhhx8GLg96379/v+2q27VUrFiRU6dO2d4fOHCA9PR02/tmzZoxf/58goKCSu1VNxERudqukykMm7OV389cxNnJwjOd6/Bk25o4Ofiq0B859KydlwHAHh4ezJgxgxkzZvxlm2rVqvH9999fdz3t27dn69at+a6xtPPx8eGhhx5izJgxpKam2oJo7dq1+fLLL9mwYQPlypVj0qRJJCYmXjcQ3XnnnUyfPp1WrVqRm5vL6NGj7b6QtX///rz77rvcd999vPHGG1SpUoWjR4+yaNEinn/+ebvxYyIiUvIZY/jfxnjeXLqbrBwrlfw9mNo3kluqBzq6tKsUm7vMxHEGDx7M+fPniYqKIjQ0FICXX36ZZs2aERUVRfv27QkJCaFHjx7XXc/7779PWFgYbdq0oV+/fjz77LN2X3rr5eXF2rVrqVq1Kj179qR+/foMHjyYjIwMXTESESllUjOyGTp3K68s2UlWjpU76wXx/dNtimUYArCYvN6nXYalpqbi7+9PSkrKVSfujIwMDh8+THh4OB4eHg6qsPTS8RURKXl2HE8hek4s8efScXGyMLpLPQa3Di/yLrLrnb//rGQMdBEREZFizxjD5xuOMO77vWTlWqkc4Mm0fpE0q1rO0aX9LQUiERERuWkp6dk8/9U2ftx1+bl/nRsE8+4DTfD3cv2bJYsHBSIRERG5KVvjzzNs7laOn7+Eq7OFF++uz6O3V7c9hLckUCASERGRG2KM4dP1h5mwbC85VkPVQC+m94ukcZUAR5eWbwpEBURj0wuHjquISPF0/mIWzy7cxsq9l78p4u6IECb0aoyfR8noIvszBaKbdOU5O+np6df8XjS5OVeejO3snL8ncIuISOH57cg5np67lZMpGbi5OPFK9wY83LJqieoi+zMFopvk7OxMQECA7bvUvLy8SvQHojixWq2cPn0aLy8vXFz0URURcTSr1fDx2kO8/9N+cq2G8AreTO8XScNQf0eXdtN0likAISEhAHZfMCsFw8nJiapVS/b/OkRESoOzaZmMWrCNn/efBuDeJqGM6xmBj3vpiBKlYy8czGKxUKlSJYKCguy+zFRunpubG05OeqC6iIgjbfz9LE/P20piaibuLk6MvbchD90S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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "ADF Statistic: -7.56194937618537\n",
            "p-value: 2.9938941433740185e-11\n",
            "Critical Values:\n",
            "   1%: -3.4370266558635914\n",
            "Critical Values:\n",
            "   5%: -2.864487711945291\n",
            "Critical Values:\n",
            "   10%: -2.5683395116993872\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.\n",
            "  self._init_dates(dates, freq)\n",
            "/usr/local/lib/python3.10/dist-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.\n",
            "  self._init_dates(dates, freq)\n",
            "/usr/local/lib/python3.10/dist-packages/statsmodels/tsa/statespace/sarimax.py:966: UserWarning: Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.\n",
            "  warn('Non-stationary starting autoregressive parameters'\n",
            "/usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:604: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals\n",
            "  warnings.warn(\"Maximum Likelihood optimization failed to \"\n",
            "/usr/local/lib/python3.10/dist-packages/statsmodels/tsa/base/tsa_model.py:834: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.\n",
            "  return get_prediction_index(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                               SARIMAX Results                                \n",
            "==============================================================================\n",
            "Dep. Variable:                  Value   No. Observations:                  999\n",
            "Model:               SARIMAX(1, 0, 0)   Log Likelihood               -2116.533\n",
            "Date:                Wed, 21 Jun 2023   AIC                           4237.067\n",
            "Time:                        10:04:25   BIC                           4246.880\n",
            "Sample:                             0   HQIC                          4240.797\n",
            "                                - 999                                         \n",
            "Covariance Type:                  opg                                         \n",
            "==============================================================================\n",
            "                 coef    std err          z      P>|z|      [0.025      0.975]\n",
            "------------------------------------------------------------------------------\n",
            "ar.L1          1.0000   2.01e-06   4.97e+05      0.000       1.000       1.000\n",
            "sigma2         3.9997   5.07e-13   7.89e+12      0.000       4.000       4.000\n",
            "===================================================================================\n",
            "Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):          41278976.44\n",
            "Prob(Q):                              0.99   Prob(JB):                         0.00\n",
            "Heteroskedasticity (H):               1.00   Skew:                           -31.56\n",
            "Prob(H) (two-sided):                  0.97   Kurtosis:                       996.83\n",
            "===================================================================================\n",
            "\n",
            "Warnings:\n",
            "[1] Covariance matrix calculated using the outer product of gradients (complex-step).\n",
            "[2] Covariance matrix is singular or near-singular, with condition number    inf. Standard errors may be unstable.\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjcAAAHHCAYAAABDUnkqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABS7UlEQVR4nO3deVhU1R8G8HcGmGHfZN8E9wVFBCE0NQvFtaxccgk1NS0zlTa11PyVopVmqWlZaouIWWpaZilumeSCoqKiuCAosokw7AMz5/cHOUXiggKXGd7P88yj3Dl35ntPMvN27zn3yIQQAkREREQGQi51AUREREQ1ieGGiIiIDArDDRERERkUhhsiIiIyKAw3REREZFAYboiIiMigMNwQERGRQWG4ISIiIoPCcENEREQGheGGiB6at7c3Ro8erft57969kMlk2Lt3r2Q1/dd/a6wrjz32GB577LE6f1+ihozhhkjPrV27FjKZTPcwNTVFixYt8MorryAjI0Pq8qpl+/btePfddyV5702bNkEmk+HLL7+8Y5udO3dCJpPh008/rcPKiKi6GG6IDMT//vc/fPvtt1i2bBk6d+6MFStWICQkBEVFRXVeS7du3VBcXIxu3bpVa7/t27dj7ty5tVTV3fXr1w82NjaIioq6Y5uoqCgYGRnhueeeq8PKiKi6GG6IDESfPn0wcuRIjBs3DmvXrsXUqVNx+fJl/PTTT3fcp7CwsFZqkcvlMDU1hVyuPx8xSqUSgwYNwr59+5CWlnbb8yUlJdi8eTN69uwJJycnCSokovulP588RFQtjz/+OADg8uXLAIDRo0fD0tISFy9eRN++fWFlZYURI0YAALRaLZYsWYK2bdvC1NQUzs7OmDBhAm7evFnpNYUQeP/99+Hh4QFzc3P06NEDp0+fvu297zTm5tChQ+jbty/s7OxgYWGB9u3b45NPPtHVt3z5cgCodJntlpqusSojR46EVqtFdHT0bc/98ssvyMvL0/XZmjVr8Pjjj8PJyQlKpRJt2rTBihUr7vkety4jJicnV9p+tz7r3bs3bGxsYG5uju7du+PPP/+s1CY/Px9Tp06Ft7c3lEolnJyc0LNnTxw7duy+jpvI0BhLXQAR1Y6LFy8CABo1aqTbVl5ejrCwMDz66KP46KOPYG5uDgCYMGEC1q5dizFjxuDVV1/F5cuXsWzZMhw/fhx//vknTExMAACzZ8/G+++/j759+6Jv3744duwYevXqBbVafc96du7cif79+8PV1RVTpkyBi4sLzp49i59//hlTpkzBhAkTkJaWhp07d+Lbb7+9bf+6qLFbt27w8PBAVFQUIiIiKj0XFRUFc3NzDBw4EACwYsUKtG3bFk8++SSMjY2xbds2vPzyy9BqtZg0adI93+t+7N69G3369EFAQADmzJkDuVyuC1V//PEHgoKCAAATJ07EDz/8gFdeeQVt2rTBjRs3cODAAZw9exYdO3askVqI9IogIr22Zs0aAUDs2rVLZGVlidTUVBEdHS0aNWokzMzMxNWrV4UQQowaNUoAENOnT6+0/x9//CEAiHXr1lXavmPHjkrbMzMzhUKhEP369RNarVbXbubMmQKAGDVqlG7bnj17BACxZ88eIYQQ5eXlwsfHRzRu3FjcvHmz0vv8+7UmTZokqvpYqo0a7+SNN94QAMS5c+d02/Ly8oSpqakYNmyYbltRUdFt+4aFhYkmTZpU2ta9e3fRvXt33c+3/ntdvny5Urv/9plWqxXNmzcXYWFhlY6lqKhI+Pj4iJ49e+q22djYiEmTJt3z2IgaCl6WIjIQoaGhcHR0hKenJ5577jlYWlpi8+bNcHd3r9TupZdeqvTzxo0bYWNjg549eyI7O1v3CAgIgKWlJfbs2QMA2LVrF9RqNSZPnlzpctHUqVPvWdvx48dx+fJlTJ06Fba2tpWe+/dr3Uld1HjLyJEjAaDSwOIff/wRJSUluktSAGBmZqb7e15eHrKzs9G9e3dcunQJeXl59/1+dxIfH4+kpCQMHz4cN27c0B1zYWEhnnjiCezfvx9arRYAYGtri0OHDlU5VoioIWrQ4Wb//v0YMGAA3NzcIJPJsGXLlmq/hhACH330EVq0aAGlUgl3d3fMmzev5osluofly5dj586d2LNnD86cOYNLly4hLCysUhtjY2N4eHhU2paUlIS8vDw4OTnB0dGx0qOgoACZmZkAgCtXrgAAmjdvXml/R0dH2NnZ3bW2W5fIfH19H+jY6qLGW9q3bw9fX1+sX79ety0qKgoODg6V+vPPP/9EaGgoLCwsYGtrC0dHR8ycORMAaiTcJCUlAQBGjRp12zF/+eWXKC0t1b3PBx98gISEBHh6eiIoKAjvvvsuLl269NA1EOmrBj3mprCwEH5+fnjhhRfwzDPPPNBrTJkyBb///js++ugjtGvXDjk5OcjJyanhSonuLSgoCIGBgXdto1Qqb5vBpNVq4eTkhHXr1lW5j6OjY43V+KDqusaRI0di+vTpOHr0KDw8PLBnzx5MmDABxsYVH5kXL17EE088gVatWmHx4sXw9PSEQqHA9u3b8fHHH+vOqFTlTmeqNBpNpZ9vvcaHH36IDh06VLmPpaUlAGDIkCHo2rUrNm/ejN9//x0ffvghFi5ciE2bNqFPnz7VPXwivdegw02fPn3u+otfWlqKt99+G+vXr0dubi58fX2xcOFC3d1Gz549ixUrViAhIQEtW7YEAPj4+NRF6UQ1pmnTpti1axe6dOlS6VLLfzVu3BhAxRmFJk2a6LZnZWXdNmOpqvcAgISEBISGht6x3Z2++Ouixn8bNmwYZsyYgaioKDRu3BgajabSJalt27ahtLQUW7duhZeXl277rctjd3PrDFJubm6l7bfOOt1yq8+sra3v2me3uLq64uWXX8bLL7+MzMxMdOzYEfPmzWO4oQapQV+WupdXXnkFsbGxiI6OxsmTJzF48GD07t1bd7p427ZtaNKkCX7++Wf4+PjA29sb48aN45kb0itDhgyBRqPBe++9d9tz5eXlui/h0NBQmJiYYOnSpRBC6NosWbLknu/RsWNH+Pj4YMmSJbd9qf/7tSwsLADc/sVfFzX+m5eXF7p27YoNGzbgu+++g4+PDzp37qx73sjI6Lba8/LysGbNmnu+9q3Qsn//ft02jUaDL774olK7gIAANG3aFB999BEKCgpue52srCzdvv+9DObk5AQ3NzeUlpbesx4iQ9Sgz9zcTUpKCtasWYOUlBS4ubkBAF5//XXs2LEDa9aswfz583Hp0iVcuXIFGzduxDfffAONRoNp06Zh0KBB2L17t8RHQHR/unfvjgkTJiAyMhLx8fHo1asXTExMkJSUhI0bN+KTTz7BoEGD4OjoiNdffx2RkZHo378/+vbti+PHj+PXX3+Fg4PDXd9DLpdjxYoVGDBgADp06IAxY8bA1dUViYmJOH36NH777TcAFV/oAPDqq68iLCxMdzfguqjxv0aOHIkXX3wRaWlpePvttys916tXLygUCgwYMAATJkxAQUEBVq1aBScnJ1y/fv2ur9u2bVs88sgjmDFjBnJycmBvb4/o6GiUl5ff1mdffvkl+vTpg7Zt22LMmDFwd3fHtWvXsGfPHlhbW2Pbtm3Iz8+Hh4cHBg0aBD8/P1haWmLXrl04cuQIFi1aVK1jJjIYks7VqkcAiM2bN+t+/vnnnwUAYWFhUelhbGwshgwZIoQQYvz48bdNGY2LixMARGJiYl0fAjVQt6YWHzly5K7tRo0aJSwsLO74/BdffCECAgKEmZmZsLKyEu3atRNvvvmmSEtL07XRaDRi7ty5wtXVVZiZmYnHHntMJCQkiMaNG991KvgtBw4cED179hRWVlbCwsJCtG/fXixdulT3fHl5uZg8ebJwdHQUMpnstmnhNVnjveTk5AilUikAiDNnztz2/NatW0X79u2Fqamp8Pb2FgsXLhSrV6++bZr3f6eCCyHExYsXRWhoqFAqlcLZ2VnMnDlT7Ny5s8o+O378uHjmmWdEo0aNhFKpFI0bNxZDhgwRMTExQgghSktLxRtvvCH8/Px0/ern5yc+++yz+z5WIkMjE+Jf51UbMJlMhs2bN+tu0LVhwwaMGDECp0+f1p2CvsXS0hIuLi6YM2cO5s+fj7KyMt1zxcXFMDc3x++//46ePXvW5SEQEREReFnqjvz9/aHRaJCZmYmuXbtW2aZLly4oLy/HxYsXddfRz58/D+CfgY1ERERUtxr0mZuCggJcuHABQEWYWbx4MXr06AF7e3t4eXlh5MiR+PPPP7Fo0SL4+/sjKysLMTExaN++Pfr16wetVotOnTrB0tISS5Ys0d123draGr///rvER0dERNQwNehws3fvXvTo0eO27aNGjcLatWtRVlaG999/H9988w2uXbsGBwcHPPLII5g7dy7atWsHAEhLS8PkyZPx+++/w8LCAn369MGiRYtgb29f14dDREREaODhhoiIiAwP73NDREREBoXhhoiIiAxKg5stpdVqkZaWBisrq/tajZiIiIikJ4RAfn4+3Nzcblsj778aXLhJS0uDp6en1GUQERHRA0hNTYWHh8dd2zS4cGNlZQWgonOsra0lroaIiIjuh0qlgqenp+57/G4aXLi5dSnK2tqa4YaIiEjP3M+QEg4oJiIiIoPCcENEREQGheGGiIiIDArDDRERERkUhhsiIiIyKAw3REREZFAYboiIiMigMNwQERGRQWG4ISIiIoPCcENEREQGheGGiIiIDArDDRERERkUhhsiIiKqOTt2AGVlkpbAcENEREQ1Y88eoE8fICQEKC6WrAyGGyIiInp4KhUwZkzF3wMCADMzyUphuCEiIqKHN20acOUK4OMDfPSRpKUw3BAREdHD2bYNWL0akMmAr78GrKwkLYfhhoiIiB5cdjYwfnzF3197DejaVdp6wHBDRERED0oIYOJEICMDaNsWeO89qSsCwHBDREREDyoqCvjxR8DYGPjmG8DUVOqKADDcEBER0YO4ehWYNKni77NnAx07SlvPvzDcEBERUfUIAYwdC+TlAUFBwIwZUldUiaThZv/+/RgwYADc3Nwgk8mwZcuWe+6zd+9edOzYEUqlEs2aNcPatWtrvU4iIiL6lxUrgN9/r7gM9fXXFZel6hFJw01hYSH8/PywfPny+2p/+fJl9OvXDz169EB8fDymTp2KcePG4bfffqvlSomIiAgAkJQEvPFGxd8XLgRatZK2nirIhBBC6iIAQCaTYfPmzRg4cOAd27z11lv45ZdfkJCQoNv23HPPITc3Fzt27Liv91GpVLCxsUFeXh6sra0ftmwiIqKGQ6OpmOodGws8/jiwcycgr5vzJNX5/tarMTexsbEIDQ2ttC0sLAyxsbESVURERNRw5P9vfkWwsbYG1qyps2BTXfWzqjtIT0+Hs7NzpW3Ozs5QqVQovsMCXaWlpVCpVJUeREREVD1CrYZq1WoAQMxLbwNeXhJXdGd6FW4eRGRkJGxsbHQPT09PqUsiIiLSO+uOXUfP5z7E7N6T4DV1otTl3JVehRsXFxdkZGRU2paRkQFra2uY3WH10RkzZiAvL0/3SE1NrYtSiYiIDMbl7ELM++UsihRm8JoZgeYu9XvMav2au3UPISEh2L59e6VtO3fuREhIyB33USqVUCqVtV0aERGRQSrXaDFtQzyKyzTo3LQRXujiI3VJ9yTpmZuCggLEx8cjPj4eQMVU7/j4eKSkpACoOOsSHh6uaz9x4kRcunQJb775JhITE/HZZ5/h+++/x7Rp06Qon4iIyOB9tvci4lNzYWVqjA8H+0Eul0ld0j1JGm6OHj0Kf39/+Pv7AwAiIiLg7++P2bNnAwCuX7+uCzoA4OPjg19++QU7d+6En58fFi1ahC+//BJhYWGS1E9ERGTITl7NxacxSQCA957yhbtt1UNA6pt6c5+busL73BAREd1bsVqD/kv/wMWsQvRr74plw/whk0l31sZg73NDREREdWPhjkRczCqEk5US8wb6ShpsqovhhoiIiCr5IykLaw8mAwA+HOwHW3OFtAVVE8MNERER6eQWqfH6xhMAgPCQxujewlHiiqqP4YaIiIh0Zv10GhmqUjRxsMCMPq2lLueBMNwQERERAOCn+GvYdiINRnIZPh7aAWYKI6lLeiAMN0RERITrecWYtSUBADD58Wbw87SVtqCHwHBDRETUwGm1Am9sPAlVSTn8PG0xqUczqUt6KAw3REREDdw3sck4cCEbpiZyfDzEDyZG+h0P9Lt6IiIieigXMvMR+WsiAODtvq3RxNFS4ooeHsMNERFRA6Uu12LqhniUlmvRrYUjRj7SWOqSagTDDRERUQO1eOd5JFxTwdbcBB8Oaq9XdyG+G4YbIiKiBujgxWx8vv8iAGDBM+3hbG0qcUU1h+GGiIiogcktUiNiwwkIAQwL8kRvXxepS6pRDDdEREQNiBACMzadQrqqBE0cLDCrfxupS6pxDDdEREQNyMa4q/g1IR3Gchk+ec4f5gpjqUuqcQw3REREDURydiHe3XoaAPBar5Zo52EjcUW1g+GGiIioASjTaDFlQzyK1Bo80sQeL3ZrInVJtYbhhoiIqAH4NCYJJ1JzYW1qjMVDOsBIbhjTvqvCcENERGTgDl/OwfI9FwAAkc+0h5utmcQV1S6GGyIiIgOWV1yGaRvioRXAoAAP9GvvKnVJtY7hhoiIyEAJIfDOlgRcyy2Gl7053n2yrdQl1QmGGyIiIgO1Jf4atp1Ig5FchiXPdYCl0vCmfVeF4YaIiMgApeYUYdaWimnfU55ojo5edhJXVHcYboiIiAxMuUaLKdHHUVBajk7edpjUo5nUJdUphhsiIiID8+nuCziWkgsrpeFP+64Kww0REZEB+evSDSzbnQQAeP9pX3jam0tcUd1juCEiIjIQNwvVmBr9z7Tvpzq4S12SJBhuiIiIDIAQAm/8cFK32vfcBjLtuyoMN0RERAbgm9gr2HU2AwojOT4d5g+LBjLtuyoMN0RERHruTJoK87afBQBM79MKvu6Gudr3/WK4ISIi0mNF6nJMXn8M6nItnmjlhDFdvKUuSXIMN0RERHps7tYzuJhVCGdrJT4c7AeZrGFN+64Kww0REZGe2nYiDRuOpkImAz4e2gH2FgqpS6oXGG6IiIj0UGpOEWZuOgUAmPRYM3Ru6iBxRfUHww0REZGeKdNoMXn9ceSXliOgsR2mhjaXuqR6heGGiIhIzyzeeR7xqbmwNjXGJ891gLERv87/jb1BRESkRw4kZWPlvosAgIXPtoeHXcNbXuFeGG6IiIj0RHZBKaZ9Hw8hgOHBXujTzlXqkuolhhsiIiI9oNUKvL7xBLLyS9HC2RKz+rWRuqR6i+GGiIhID3zxxyXsPZcFpbEcS4d1hJnCSOqS6i2GGyIionruaHIOPvztHADg3SfboqWLlcQV1W8MN0RERPXYzUI1Jq8/Do1W4KkObniuk6fUJdV7DDdERET1lFYr8NrGE7ieV4ImDhaY93Q7Lq9wHxhuiIiI6qkvD1zC7sRMKIzlWDa8IyyVxlKXpBcYboiIiOqhuCs5WLjj73E2A9qijZu1xBXpD4YbIiKieuZmoRqToyrG2Tzp54ZhQRxnUx0MN0RERPWIEBX3s0nLK4GPgwXmP8NxNtXFcENERFSPfPnHZcToxtn4c5zNA2C4ISIiqieOpdzEwh2JAIA5A9qgrZuNxBXpJ4YbIiKieiC3qGKcTblWYICfG4YHeUldkt5iuCEiIpLYrXE213KL4d3IHPOf9uU4m4fAcENERCSxrw5cxq6z/9zPxsrUROqS9BrDDRERkYSOp9zEgl8rxtnM7t8Gvu4cZ/OwGG6IiIgkcrNQjUnrjqFcK9C/vStGBHOcTU2QPNwsX74c3t7eMDU1RXBwMA4fPnzX9kuWLEHLli1hZmYGT09PTJs2DSUlJXVULRERUc3QagWmbojX3c8mkvezqTGShpsNGzYgIiICc+bMwbFjx+Dn54ewsDBkZmZW2T4qKgrTp0/HnDlzcPbsWXz11VfYsGEDZs6cWceVExERPZyluy9g3/ksmJrIsWIkx9nUJEnDzeLFizF+/HiMGTMGbdq0wcqVK2Fubo7Vq1dX2f7gwYPo0qULhg8fDm9vb/Tq1QvDhg2759keIiKi+mT/+SwsiTkPAJj/dDu0cuG6UTVJsnCjVqsRFxeH0NDQf4qRyxEaGorY2Ngq9+ncuTPi4uJ0YebSpUvYvn07+vbte8f3KS0thUqlqvQgIiKSSlpuMaZEH4cQwLAgLzzT0UPqkgyOZPd0zs7OhkajgbOzc6Xtzs7OSExMrHKf4cOHIzs7G48++iiEECgvL8fEiRPvelkqMjISc+fOrdHaiYiIHoS6XIuX1x3DzaIy+LpbY86ANlKXZJAkH1BcHXv37sX8+fPx2Wef4dixY9i0aRN++eUXvPfee3fcZ8aMGcjLy9M9UlNT67BiIiKif8zffhbxqbmwMTPBihEBMDUxkrokgyTZmRsHBwcYGRkhIyOj0vaMjAy4uLhUuc+sWbPw/PPPY9y4cQCAdu3aobCwEC+++CLefvttyOW3ZzWlUgmlUlnzB0BERFQNW0+kYe3BZADAx0P94GlvLm1BBkyyMzcKhQIBAQGIiYnRbdNqtYiJiUFISEiV+xQVFd0WYIyMKlKvEKL2iiUiInoIFzLzMf3HkwCAST2a4vFWzvfYgx6GpOuoR0REYNSoUQgMDERQUBCWLFmCwsJCjBkzBgAQHh4Od3d3REZGAgAGDBiAxYsXw9/fH8HBwbhw4QJmzZqFAQMG6EIOERFRfVJYWo6J3x1DkVqDzk0bIaJnS6lLMniShpuhQ4ciKysLs2fPRnp6Ojp06IAdO3boBhmnpKRUOlPzzjvvQCaT4Z133sG1a9fg6OiIAQMGYN68eVIdAhER0R0JITBj0ylcyCyAs7USnw7zh5GcN+qrbTLRwK7nqFQq2NjYIC8vD9bWvK8AERHVnm9ikzH7p9MwkssQ/eIj6ORtL3VJeqs63996NVuKiIhIXxxPuYn3fj4DAJjRpxWDTR1iuCEiIqphOX8viFmmEejj64Kxj/pIXVKDwnBDRERUgzRagVfXH9ctiPnBoPZcELOOMdwQERHVoA9/O4cDF7JhZmKElSMDuCCmBBhuiIiIasivp65j5b6LAIAPBrVHSxcriStqmBhuiIiIakBSRj5e33gCADC+qw8G+LlJXFHDxXBDRET0kFQlZZjwbRwK1RqENGmEt3q3krqkBo3hhoiI6CFotQKvfX8Cl7IL4WZjimXD/WFsxK9XKbH3iYiIHsJney9g55kMKIzkWDEyAI0suViz1BhuiIiIHtDec5lYtPM8AOC9gW3h52krbUEEgOGGiIjogaTcKMKU6HgIAQwL8sLQTl5Sl0R/Y7ghIiKqpmK1BhO+i0NecRk6eNri3SfbSF0S/QvDDRERUTVUrPR9Emevq+BgqcCKkR2hNDaSuiz6F4YbIiKialh7MBlb4tNgJJdh2fCOcLUxk7ok+g+GGyIiovt06NINzPvlLABgZt/WeKRJI4kroqow3BAREd2Ha7nFeHndMZRrBZ70c8MLXbylLonugOGGiIjoHorVGrz4zVHcKFSjrZs1Fj7Llb7rM4YbIiKiuxBC4M0fT+J0mgr2Fgp8/nwAzBQcQFyfMdwQERHdxef7L2HbiTQYy2X4bERHeNiZS10S3QPDDRER0R3sSczEwh2JAIA5T7blAGI9wXBDRERUhUtZBXg1+vjfdyD2xMhg3oFYXzDcEBER/YeqpAzjvzmK/JJyBDa2w9wnfTmAWI8w3BAREf2LRiswNToeF7MK4WpjihUjA6Aw5telPuF/LSIion9ZvPMcdidmQmksx+fPB8DRSil1SVRNDDdERER/+/lkGpbvuQgAWPBsO7T3sJW2IHogDDdEREQATqfl4Y2NJwEAL3Zrgqf9PSSuiB4Uww0RETV4NwpK8eI3cSgu06Brcwe81buV1CXRQ2C4ISKiBk1drsXL647hWm4xvBuZY9mwjjCSc2aUPmO4ISKiBksIgVlbEnDocg4slcZYFR4IG3MTqcuih8RwQ0REDdZXBy5jw9FUyGXA0uH+aO5sJXVJVAMYboiIqEHak5iJ+dvPAgBm9m2NHi2dJK6IagrDDRERNTjnM/Ixef1xaAXwXCdPjH3UR+qSqAYx3BARUYOSU6jG2K+PoKC0HME+9vjfU1xawdAw3BARUYOhLtdi4rdxSM0phpe9OZdWMFD8L0pERA2CEALvbDmFw8k5sFIa46tRgbC3UEhdFtUChhsiImoQvjpwGd8fvQq5DPiUM6MMGsMNEREZvN2JGZj398yod/q14cwoA8dwQ0REBu1cej5eXR8PIYBhQZ4Y08Vb6pKoljHcEBGRwbpRUKqbGfVIE3vMfZIzoxoChhsiIjJIpeUaTPwuDldvFqNxI3OsGMGZUQ0F/ysTEZHBEULgzR9O4kjyTd3MKDvOjGowGG6IiMjgfLwrCT/Fp8FYLsOKkQFo5sSZUQ0Jww0RERmUH+Ou4tOYJADA+wN98WhzB4krorrGcENERAbjr0s3MH3TSQDAS481xXNBXhJXRFJguCEiIoNwMasAE76NQ5lGoF87V7zRq6XUJZFEGG6IiEjv3SgoxZg1R5BXXAZ/L1ssGuIHuZxTvhsqhhsiItJrJWUavPhtHFJyiuBhZ4ZV4YEwNTGSuiySEMMNERHpLa1W4I0fTiLuyk1YmRpj7ZhOcLBUSl0WSYzhhoiI9NbHu85j24mKKd+fc8o3/Y3hhoiI9NLGo6lYuvsCAGD+0+3QuRmnfFMFhhsiItI7By9mY8amUwCAST2aYkgnT4krovqE4YaIiPRKUkY+Jn4bh3KtQP/2rnitJ6d8U2UMN0REpDcyVCUYveYIVCXl6Ohli48Gc8o33Y7hhoiI9EJ+SRlGrzmCa7nFaOJggS9HdeKUb6qS5OFm+fLl8Pb2hqmpKYKDg3H48OG7ts/NzcWkSZPg6uoKpVKJFi1aYPv27XVULRERSUFdrsVL3x3D2esqOFgqsHZMEOy5yjfdgbGUb75hwwZERERg5cqVCA4OxpIlSxAWFoZz587BycnptvZqtRo9e/aEk5MTfvjhB7i7u+PKlSuwtbWt++KJiKhOCCEwfdNJHLiQDXOFEVaP7gSvRuZSl0X1mEwIIaR68+DgYHTq1AnLli0DAGi1Wnh6emLy5MmYPn36be1XrlyJDz/8EImJiTAxMXmg91SpVLCxsUFeXh6sra0fqn4iIqp9i34/h6W7L8BILsOX4YHo0er2//klw1ed72/JLkup1WrExcUhNDT0n2LkcoSGhiI2NrbKfbZu3YqQkBBMmjQJzs7O8PX1xfz586HRaO74PqWlpVCpVJUeRESkH6IOpejuZTNvoC+DDd0XycJNdnY2NBoNnJ2dK213dnZGenp6lftcunQJP/zwAzQaDbZv345Zs2Zh0aJFeP/99+/4PpGRkbCxsdE9PD15LwQiIn0QczYD72ypuJfNq080x3NBXhJXRPpC8gHF1aHVauHk5IQvvvgCAQEBGDp0KN5++22sXLnyjvvMmDEDeXl5ukdqamodVkxERA/iRGouXok6Dq0ABgd4YFpoc6lLIj0i2YBiBwcHGBkZISMjo9L2jIwMuLi4VLmPq6srTExMYGT0z9S/1q1bIz09HWq1GgrF7SPnlUollEouokZEpC+u3CjEC2uPoLhMg24tHDH/mXaQyXgvG7p/kp25USgUCAgIQExMjG6bVqtFTEwMQkJCqtynS5cuuHDhArRarW7b+fPn4erqWmWwISIi/XKjoBSj1xzBjUI12rpZ47MRHWFipFcXGagekPRfTEREBFatWoWvv/4aZ8+exUsvvYTCwkKMGTMGABAeHo4ZM2bo2r/00kvIycnBlClTcP78efzyyy+YP38+Jk2aJNUhEBFRDSlWazDum6O4nF0Id1szrBndCZZKSe9YQnpK0n81Q4cORVZWFmbPno309HR06NABO3bs0A0yTklJgVz+T/7y9PTEb7/9hmnTpqF9+/Zwd3fHlClT8NZbb0l1CEREVAPKNFpMijqG4ym5sDEzwdcvdIKTtanUZZGekvQ+N1LgfW6IiOoXIQRe33gSPx67CqWxHOvGBSPQ217qsqie0Yv73BAREQHAgl8T8eOxqzCSy/DZiI4MNvTQGG6IiEgyq/Zfwuf7LwEAFjzTDk+0dr7HHkT39kDhpry8HLt27cLnn3+O/Px8AEBaWhoKCgpqtDgiIjJcP8ZdxbztZwEA0/u0wuBA3mSVaka1BxRfuXIFvXv3RkpKCkpLS9GzZ09YWVlh4cKFKC0tvesN9YiIiABgT2Im3vzxJABg3KM+mNCticQVkSGp9pmbKVOmIDAwEDdv3oSZmZlu+9NPP13pnjVERERVibtyEy+ti4NGK/CMvztm9m3Nm/RRjar2mZs//vgDBw8evO2med7e3rh27VqNFUZERIbnfEY+Xlh7BCVlWvRo6YiFg9pDLmewoZpV7TM3Wq22ylW4r169CisrqxopioiIDM+13GKEf3UYecVl8PeyxXLefZhqSbX/VfXq1QtLlizR/SyTyVBQUIA5c+agb9++NVkbEREZiJxCNcK/OoR0VQmaOVli9ahOMFfw7sNUO6p9E7+rV68iLCwMQggkJSUhMDAQSUlJcHBwwP79++Hk5FRbtdYI3sSPiKhuFanLMXzVIcSn5sLVxhQ/vtQZbrZm996R6F+q8/1d7djs4eGBEydOIDo6GidPnkRBQQHGjh2LESNGVBpgTEREVFquwYRv4xCfmgtbcxN8OzaIwYZq3QOdEzQ2NsbIkSNruhYiIjIg5RotpqyPxx9J2TBXGGHN6E5o5sSxmVT7qh1uvvnmm7s+Hx4e/sDFEBGRYdBqBd768RR2nE6HwkiOVeGB8Peyk7osaiCqPebGzq7yP86ysjIUFRVBoVDA3NwcOTk5NVpgTeOYGyKi2iWEwNxtZ7D2YDKM5DKsGNERvdq6SF0W6blaXTjz5s2blR4FBQU4d+4cHn30Uaxfv/6BiyYiIsPw8c7zWHswGQDw0eD2DDZU52rkBgPNmzfHggULMGXKlJp4OSIi0lOr9l/Cp7svAADee6otnvb3kLgiaohq7O5JxsbGSEtLq6mXIyIiPRN9OEW3EOYbYS3xfIi3tAVRg1XtAcVbt26t9LMQAtevX8eyZcvQpUuXGiuMiIj0x88n0zBj8ykAwITuTfDyY00lrogasmqHm4EDB1b6WSaTwdHREY8//jgWLVpUU3UREZGe2JOYianR8RACGB7shem9W3EhTJJUtcONVqutjTqIiEgPHbp0AxO/i0O5VuBJPze895Qvgw1JjiuWERHRAzmRmouxXx9FabkWj7dywqIhfjDiCt9UD9zXmZuIiIj7fsHFixc/cDFERKQfTqflIXz1YRSUluORJvb4jCt8Uz1yX+Hm+PHj9/ViPBVJRGT4zmfk4/mvDiOvuAwBje3w1ahOMDUxkrosIp37Cjd79uyp7TqIiEgPXMoqwPBVh5BTqEZ7DxusGdMJFsoHWqaQqNbwHCIREd2XlBtFGL7qELILStHa1RrfvBAEa1MTqcsius0Dxe2jR4/i+++/R0pKCtRqdaXnNm3aVCOFERFR/XEttxjDVv2FdFUJmjtZ4ruxQbA1V0hdFlGVqn3mJjo6Gp07d8bZs2exefNmlJWV4fTp09i9ezdsbGxqo0YiIpJQhqoEw1f9hWu5xfBxsMC6ccFoZKmUuiyiO6p2uJk/fz4+/vhjbNu2DQqFAp988gkSExMxZMgQeHl51UaNREQkkeyCUgxf9Reu3CiCp70ZosYHw8naVOqyiO6q2uHm4sWL6NevHwBAoVCgsLAQMpkM06ZNwxdffFHjBRIRkTRuFqox8stDuJhVCDcbU0SNewSuNmZSl0V0T9UON3Z2dsjPzwcAuLu7IyEhAQCQm5uLoqKimq2OiIgkkVdchudXH0Jiej6crJRYN/4ReNqbS10W0X2573BzK8R069YNO3fuBAAMHjwYU6ZMwfjx4zFs2DA88cQTtVMlERHVGVVJGUatPoyEayo0slAganwwfBwspC6L6L7d92yp9u3bo1OnThg4cCAGDx4MAHj77bdhYmKCgwcP4tlnn8U777xTa4USEVHtuxVs4lNzYWtugu/GBaOZk5XUZRFVi0wIIe6n4R9//IE1a9bghx9+gFarxbPPPotx48aha9eutV1jjVKpVLCxsUFeXh6sra2lLoeIqN7ILylD+OrDOJ7yd7AZGwxfd86CpfqhOt/f931ZqmvXrli9ejWuX7+OpUuXIjk5Gd27d0eLFi2wcOFCpKenP3ThREQkDQYbMiTVHlBsYWGBMWPGYN++fTh//jwGDx6M5cuXw8vLC08++WRt1EhERLXo38HGxozBhvTffV+WupPCwkKsW7cOM2bMQG5uLjQaTU3VVit4WYqI6B//DTbrxjHYUP1Une/vB17tbP/+/Vi9ejV+/PFHyOVyDBkyBGPHjn3QlyMiojqW//fgYQYbMjTVCjdpaWlYu3Yt1q5diwsXLqBz58749NNPMWTIEFhYcJogEZG+uBVsjjHYkAG673DTp08f7Nq1Cw4ODggPD8cLL7yAli1b1mZtRERUC/JLyjB6zREGGzJY9x1uTExM8MMPP6B///4wMjKqzZqIiKiW3Ao2cVduMtiQwbrvcLN169barIOIiGpZXnEZRq/hGBsyfA88oJiIiPTHzUI1wlcfxqlreZzuTQaP4YaIyMBlF5Ri5JcVi2A2slDg27HBaOPGW2GQ4WK4ISIyYBmqEoz48hAuZBbA0UqJqHHBaO7MtaLIsDHcEBEZqLTcYgxf9ReSbxTB1cYUUeMf4ere1CAw3BARGaDUnCIMW/UXrt4shoedGdaPfwSe9uZSl0VUJxhuiIgMzKWsAoz48hCu55XAu5E5osY/AjdbM6nLIqozDDdERAYkKSMfw788hKz8UjRzskTUuGA4WZtKXRZRnWK4ISIyEGfSVBj51SHkFKrRysUK340LhoOlUuqyiOocww0RkQE4kZqL8NWHkVdcBl93a3z7QjDsLBRSl0UkCYYbIiI9d/BiNsZ/fRSFag06eNri6xeCYGNmInVZRJJhuCEi0mO7zmTg5ahjUJdr0blpI3wRHghLJT/aqWHjbwARkZ76Kf4aIr4/AY1WILS1M5YN94epCRc2JmK4ISLSQ9/+dQWzf0qAEMDT/u74YFB7mBjJpS6LqF6oF78Jy5cvh7e3N0xNTREcHIzDhw/f137R0dGQyWQYOHBg7RZIRFSPfLb3AmZtqQg24SGNsWiwH4MN0b9I/tuwYcMGREREYM6cOTh27Bj8/PwQFhaGzMzMu+6XnJyM119/HV27dq2jSomIpCWEwIJfE/HBjnMAgFd6NMPcJ9tCLpdJXBlR/SJ5uFm8eDHGjx+PMWPGoE2bNli5ciXMzc2xevXqO+6j0WgwYsQIzJ07F02aNKnDaomIpKHRCry9JQEr910EAMzo0wqvh7WETMZgQ/RfkoYbtVqNuLg4hIaG6rbJ5XKEhoYiNjb2jvv973//g5OTE8aOHVsXZRIRSapMo8W0DfGIOpQCmQyIfKYdJnRvKnVZRPWWpAOKs7OzodFo4OzsXGm7s7MzEhMTq9znwIED+OqrrxAfH39f71FaWorS0lLdzyqV6oHrJSKqa8VqDSZFHcPuxEwYy2X4eGgHDPBzk7osonpN8stS1ZGfn4/nn38eq1atgoODw33tExkZCRsbG93D09OzlqskIqoZuUVqjPzqEHYnZkJpLMeq8EAGG6L7IOmZGwcHBxgZGSEjI6PS9oyMDLi4uNzW/uLFi0hOTsaAAQN027RaLQDA2NgY586dQ9OmlU/VzpgxAxEREbqfVSoVAw4R1XtpucUYtfowkjILYG1qjK9Gd0Inb3upyyLSC5KGG4VCgYCAAMTExOimc2u1WsTExOCVV165rX2rVq1w6tSpStveeecd5Ofn45NPPqkytCiVSiiVXDiOiPRHUkY+wlcfxvW8ErhYm+LrF4LQ0sVK6rKI9IbkN/GLiIjAqFGjEBgYiKCgICxZsgSFhYUYM2YMACA8PBzu7u6IjIyEqakpfH19K+1va2sLALdtJyLSR3FXbuKFtUeQV1yGJo4W+HZsMNxtzaQui0ivSB5uhg4diqysLMyePRvp6eno0KEDduzYoRtknJKSArlcr4YGERE9kJizGZgUdQwlZVp08LTF6tGdYM+VvYmqTSaEEFIXUZdUKhVsbGyQl5cHa2trqcshIgIAfH80FTM2nYJGK9CjpSOWj+gIc4Xk//9JVG9U5/ubvzlERBISQmDFvou6uw4/29EDC55tx+UUiB4Cww0RkUS0WoH3fjmDNX8mAwAmdm+Kt3rzrsNED4vhhohIAiVlGry28QR+OXkdADCrfxuMfdRH4qqIDAPDDRFRHbtZqMb4b47i6JWbMDGS4aPBfniqg7vUZREZDIYbIqI6lHKjCKPXHMal7EJYmRrj8+cD0Lnp/d1xnYjuD8MNEVEdiU/Nxdi1R3CjUA13WzOsGdMJLZx5cz6imsZwQ0RUB34/nY5Xo4+jpEyLtm7WWDO6E5ysTaUui8ggMdwQEdWyrw8m491tpyEE8FhLRywf3hEWSn78EtUW/nYREdUSrVYg8tezWPXHZQDAsCBPvPeUL4x5DxuiWsVwQ0RUC0rKNHjt+xP45VTFVO83wlri5cea8h42RHWA4YaIqIZlF5RiwrdxiONUbyJJMNwQEdWgc+n5GPv1EVy9WQxrU2N8/nwgQpo2krosogaF4YaIqIbsPZeJV6KOo6C0HN6NzPHV6E5o6mgpdVlEDQ7DDRFRDfj6YDLmbjsNrQCCfeyxcmQA7CwUUpdF1CAx3BARPYRyjRb/+/kMvom9AgAYHOCBeU+3g8KYM6KIpMJwQ0T0gFQlZXgl6jj2n8+CTAa81bsVJnRrwhlRRBJjuCEiegApN4ow9usjSMosgJmJEZY81wFhbV2kLouIwHBDRFRtR5Nz8OK3ccgpVMPZWomvRnWCr7uN1GUR0d8YboiIquH7o6l4Z3MC1Bot2rnbYFV4IFxsuEYUUX3CcENEdB/KNFrM++Us1h5MBgD08XXBoiF+MFfwY5SovuFvJRHRPdwsVGNS1DEcvHgDABDRswVe6dEMcjkHDhPVRww3RER3kZiuwvhvjiI1pxgWCiMsHsqBw0T1HcMNEdEd/HrqOl7beAJFag287M2xKjwQLV2spC6LiO6B4YaI6D+0WoElMUn4NCYJAPBoMwcsG+4PW3PecZhIHzDcEBH9S0FpOaZtiMfOMxkAgBe6+GBm31YwNuIdh4n0BcMNEdHfLmcXYsK3R3E+owAKIznmPe2LwYGeUpdFRNXEcENEBGDnmQxEbIhHfmk5nKyU+Pz5APh72UldFhE9AIYbImrQNFqBj3eex7I9FwAAAY3t8NmIjnC25o35iPQVww0RNVg3C9V4Nfo4/kjKBgCM7uyNmX1bc0VvIj3HcENEDdKpq3mY+F0cruUWw9REjgXPtMdAf3epyyKiGsBwQ0QNzoYjKZj102moy7Vo3MgcK0cGoLWrtdRlEVENYbghogajtFyDd7eexvrDqQCA0NZOWDSkA2zMTCSujIhqEsMNETUI13KL8fJ3cThxNQ8yGfBazxZ4+TGuD0VkiBhuiMjgxZzNwGsbTyC3qAy25ib45Dl/dG/hKHVZRFRLGG6IyGCVabT46Ldz+Hz/JQBAew8bLB/eEZ725hJXRkS1ieGGiAxSWm4xJq8/jrgrNwFUTPOe0bcVlMZGEldGRLWN4YaIDM6exExEfB+Pm0VlsDI1xoeD2qO3r6vUZRFRHWG4ISKDUa7R4qPfz2PlvosAgHbuFZehvBrxMhRRQ8JwQ0QGIT2vBJPXH8OR5IrLUKNCGmNmv9a8DEXUADHcEJHe23suExHfn0BOoRpWSmMsHNQefdvxMhRRQ8VwQ0R6q7Rcgw93nMOXBy4DANq6WWP58I7wdrCQuDIikhLDDRHppYtZBXh1/XGcTlMBAMJDGmNm39YwNeFlKKKGjuGGiPSKEALfH03Fu1vPoLhMAztzE3w4yA+hbZylLo2I6gmGGyLSG3lFZZi5+RR+OXUdANClWSMsHtIBztamEldGRPUJww0R6YUjyTmYGh2Pa7nFMJbL8FqvlpjQrQnXhiKi2zDcEFG9Vq7RYunuC1i6OwlaATRuZI5Pn/OHn6et1KURUT3FcENE9VZqThGmbYjH0b+XUHimozv+95QvLJX86CKiO+MnBBHVO7cGDf9v2xkUqjWwUhrj/ad98VQHd6lLIyI9wHBDRPVKVn4pZmw6iV1nMwEAnbztsHhIB67kTUT3jeGGiOqNHQnpmLn5FHIK1VAYyRHRqwXGd20CIw4aJqJqYLghIsnll5Rh7rYz+CHuKgCglYsVPh7aAa1drSWujIj0EcMNEUnqr0s38Nr3J3AttxgyGTChW1NM69mcC14S0QNjuCEiSZSUabDo94p1oYQAPO3NsHhIB3Tytpe6NCLScww3RFTn4q7k4I2NJ3EpuxAAMDTQE7MGtOEUbyKqEXKpCwCA5cuXw9vbG6ampggODsbhw4fv2HbVqlXo2rUr7OzsYGdnh9DQ0Lu2J6L6o1itwXs/n8GglbG4lF0IJyslvgwPxMJB7RlsiKjGSB5uNmzYgIiICMyZMwfHjh2Dn58fwsLCkJmZWWX7vXv3YtiwYdizZw9iY2Ph6emJXr164dq1a3VcORFVx+HLOejzyX589fdlqEEBHtg5rTsXvCSiGicTQggpCwgODkanTp2wbNkyAIBWq4WnpycmT56M6dOn33N/jUYDOzs7LFu2DOHh4fdsr1KpYGNjg7y8PFhbcyYGUW0rUpfjgx3n8HVsMoQAXKxNEflMO/Ro5SR1aUSkR6rz/S3peWC1Wo24uDjMmDFDt00ulyM0NBSxsbH39RpFRUUoKyuDvX3VgxBLS0tRWlqq+1mlUj1c0UR032Iv3sBbP55ESk4RgIqxNW/3bw1rUxOJKyMiQyZpuMnOzoZGo4Gzc+XT0s7OzkhMTLyv13jrrbfg5uaG0NDQKp+PjIzE3LlzH7pWIrp/+SVl+GDHOXz71xUAgJuNKSKfbY/uLRwlroyIGgK9HsG3YMECREdHY+/evTA1Na2yzYwZMxAREaH7WaVSwdPTs65KJGpwfjudjtk/JSBDVXHGdFiQF2b2bQUrnq0hojoiabhxcHCAkZERMjIyKm3PyMiAi4vLXff96KOPsGDBAuzatQvt27e/YzulUgmlUlkj9RLRnaXnlWDO1gT8drri99m7kTnmP90OnZs5SFwZETU0ks6WUigUCAgIQExMjG6bVqtFTEwMQkJC7rjfBx98gPfeew87duxAYGBgXZRKRHeg1Qp8G5uM0MX78NvpDBjLZZjUoyl2TO3GYENEkpD8slRERARGjRqFwMBABAUFYcmSJSgsLMSYMWMAAOHh4XB3d0dkZCQAYOHChZg9ezaioqLg7e2N9PR0AIClpSUsLS0lOw6ihuhcej5mbDqJYym5AIAOnrZY8Gw7tHLhTEQiko7k4Wbo0KHIysrC7NmzkZ6ejg4dOmDHjh26QcYpKSmQy/85wbRixQqo1WoMGjSo0uvMmTMH7777bl2WTtRglZRpsGz3BazcdxHlWgFLpTHe7N0SI4IbcwVvIpKc5Pe5qWu8zw3Rw9l7LhPvbj2N5BsV07t7tnHG/55qC1cbM4krIyJDpjf3uSEi/XEttxj/23ZaN2DY2VqJuU/6orfv3Qf/ExHVNYYbIrordbkWq/64hKW7k1BSpoWRXIYXunhjSmgLrgdFRPUSP5mI6I4OJGVj9tYEXMqqWL07yMce7z3li5YuVhJXRkR0Zww3RHSb63nFeP+Xs/jl5HUAgIOlEm/3a4WBHdwhk3HAMBHVbww3RKRTWq7B6gPJWLo7CUVqDeQyIDzEG9N6toCNGe8wTET6geGGiCCEwO9nMjDvl7O6RS47etnivYG+aOtmI3F1RETVw3BD1MAlpqvwv21ncPDiDQCAk5USb/ZuhWf83SHnPWuISA8x3BA1UDcKSrF453msP5wCrQAUxnKM7+qDlx9rBgvOgiIiPcZPMKIGRl2uxTexyfgkJgn5JeUAgL7tXDCjT2t42ptLXB0R0cNjuCFqIIQQ2HU2E5Hbz+JSdsXU7tau1pgzoA0eadJI4uqIiGoOww1RA3A85SYityficHIOAKCRhQKvh7XEkEBPrgVFRAaH4YbIgF3OLsSHvyVi+6l0AIDSWI4XHvXBS481hbUpp3YTkWFiuCEyQNkFpfg0JglRh1JQrhWQyYBBHT0Q0asFF7gkIoPHcENkQIrU5fjyj8v4fN9FFKo1AIAeLR3xVp9WaOVy91V0iYgMBcMNkQFQl2ux4UgKlu6+gMz8UgBAO3cbzOjbCp2bOkhcHRFR3WK4IdJj5RotNh2/hk92JeFabjEAwNPeDG+EtUL/dq68CR8RNUgMN0R6SKsV2HYyDUt2JeHy39O6nayUmPx4Mwzp5AmlsZHEFRIRSYfhhkiPCCHw2+kMfLzzPM5l5AMA7C0UeKl7U4x8pDHMFAw1REQMN0R6QAiBfeezsOj38zh1LQ8AYGVqjAndmmB0Fx9YcrkEIiIdfiIS1WO37iq8bHcSTlytCDXmCiO80MUH47s2gY0571VDRPRfDDdE9ZBWK7DjdDqW7r6As9dVAABTEzlGBjfGS481RSNLpcQVEhHVXww3RPVIuUaLn09ex7I9F3AhswAAYKEwwvMh3hjX1QcODDVERPfEcENUD5RptNh87Bo+23sByTeKAFSMqRnTxQcvdPGGrblC4gqJiPQHww2RhApLy7HhSCq+OnBZd58aO3MTjOvaBM+HNOb6T0RED4DhhkgCWfml+PpgMr796wryissAAA6WSrzYzQcjghvDgrOfiIgeGD9BierQxawCfPnHJfx47BrU5VoAgI+DBcZ19cGzHT1gasL71BARPSyGG6I6EHclB5/vu4SdZzMgRMU2fy9bTOjWFD3bOMOIyyQQEdUYhhuiWqIu1+LXhOtY82cy4lNzddtDWztjQvcmCGxsB5mMoYaIqKYx3BDVsOyCUkQdSsF3f13RrdCtMJLjaX93jO/mg2ZOVhJXSERk2BhuiGpIwrU8rPkzGdtOpEGtqRhP42SlxMhHGmN4sBfvUUNEVEcYbogegrpci99Op+Ob2GQcSb6p297B0xZjunijj68rFMZyCSskImp4GG6IHsCVG4VYfzgVG4+m4kahGgBgYiRDv3auGNXZG/5edhJXSETUcDHcEN2nMo0Wu85kIOpwCv5IytZtd7ZWYmgnL4wM9oKTtamEFRIREcBwQ3RPV28WIfpwKjYcTUXW3wOEZTKgW3NHDA/2whOtnGBsxEtPRET1BcMNURWK1Rr8djodP8RdxZ8Xs3X3pnGwVGJIoAeGBXnB095c2iKJiKhKDDdEfxNCIO7KTfwQdxU/n7yOgtJy3XNdmjXCiODGCG3tzAHCRET1HMMNNXjXcoux+dhV/BB3VbciNwB42pvh2Y4eeLajB8/SEBHpEYYbapByi9T4NSEdW+PT8NflG7rLTuYKI/Rr54pBAR7o5G0POZdFICLSOww31GAUlpZj19kMbI1Pw/6kLJRphO65kCaNMCjAA719XbgiNxGRnuOnOBm00nIN9p3LwtYTadh1NgMlZVrdc61drfGknxsG+LnCw46XnYiIDAXDDRmcInU59p/Pwo6EdMQkZiK/5J+Bwd6NzPGknxue7ODGNZ6IiAwUww0ZhLyiMsQkZuC30+nYdz6r0hkaF2tT9G/viic7uKGduw1X4iYiMnAMN6S30vNKEJOYgR0J6Yi9eAPl2n/G0Hjam6F3Wxf09nWBv6cdBwYTETUgDDekNzRagfjUXOxJzMTuxEycua6q9HxLZyuE+bogrK0z2rha8wwNEVEDxXBD9VpukRr7zmdh77ks7DufhZy/F6kEKpZA8POwRVjbikDTxNFSwkqJiKi+YLihekVdrkV8ai7+vJCNPy9k41jKTfzrahOsTI3RrYUjHm/phO4tHeFgqZSuWCIiqpcYbkhSWq1AYno+Dl7MxoEL2Th8OQdFak2lNi2cLdGjlRMeb+mEgMZ2XKSSiIjuiuGG6pRWK3A+Mx9HLufg0OUcxF68gRv/utQEAI0sFAhp2giPNnNAl2YOXPqAiIiqheGGalVpuQanrubhcHIOjibfxNHkHKj+dd8ZADAzMUJwE3t0aVoRZlq5WHF2ExERPTCGG6oxQghczyvBidRcnLiah2NXbiL+ai7U5dpK7cwVRujoZYdAbzuENGkEfy87rrRNREQ1huGGHlhukRonrubhRGouTl7NRXxqHrILSm9r18hCgU7e9gj0tkOQjz1au1rDhONmiIioljDc0D1ptQKpN4tw9no+EtNVOHtdhbPX85GSU3RbWyO5DC2dreDnaYMOnrYI9LZHEwcL3nOGiIjqDMMN6QghkFVQiouZhbiQVYCz11VIvK7CufR8FP5nBtMtPg4WaO9hAz8PW/h52qCNqw3MFEZ1XDkREdE/GG4aoJIyDZJvFOJSViEuZRXg4t9/XsoqRH5peZX7KIzlaOFsiVYu1mjtao3WLlZo62YDG3OTOq6eiIjo7upFuFm+fDk+/PBDpKenw8/PD0uXLkVQUNAd22/cuBGzZs1CcnIymjdvjoULF6Jv3751WHH9VlKmQVpuMa7erHik3iz6++8Vf2bl3z4u5ha5DPCwM0dTRwu0cv0nyPg4WPD+MkREpBckDzcbNmxAREQEVq5cieDgYCxZsgRhYWE4d+4cnJycbmt/8OBBDBs2DJGRkejfvz+ioqIwcOBAHDt2DL6+vhIcQd0o12iRV1yG3OIy3ChQIzO/BJmqUmTmlyIzvwRZ+aV//1yCm0Vl93w9a1NjNHG0RBNHCzR1tERTRws0cbRE40bmUBrzshIREekvmRBC3LtZ7QkODkanTp2wbNkyAIBWq4WnpycmT56M6dOn39Z+6NChKCwsxM8//6zb9sgjj6BDhw5YuXLlPd9PpVLBxsYGeXl5sLa2rrHjKC3XILtADY1GoFyrhVYIlGsFyjUCGq2ARlT8WVauRZFag6IyDYrV5SjW/V1TsV2tgaq4DLnF6oowU1SGvKKyO14uuhNzhRE87czhYWf296Pi7572FX/amJlwkC8REemN6nx/S3rmRq1WIy4uDjNmzNBtk8vlCA0NRWxsbJX7xMbGIiIiotK2sLAwbNmypcr2paWlKC395zKMSqWqst3DSriWh2dXVF1zTbI2NYa9hQJOVqZwtFbCyUoJJyvTij+tlXC0UsLZyhS25gwvRETUMEkabrKzs6HRaODs7Fxpu7OzMxITE6vcJz09vcr26enpVbaPjIzE3Llza6bguzCWy6EwlsNYLoPR3w9j3Z/yStvMFUYwUxjBXGEMM4URzEyM/tlmYgwbM2PYmitgY24CWzMT2JorYGtmAmszExjxzr1ERER3JfmYm9o2Y8aMSmd6VCoVPD09a/x9/Dxtcf79PjX+ukRERFQ9koYbBwcHGBkZISMjo9L2jIwMuLi4VLmPi4tLtdorlUoolcqaKZiIiIjqPUnn9ioUCgQEBCAmJka3TavVIiYmBiEhIVXuExISUqk9AOzcufOO7YmIiKhhkfyyVEREBEaNGoXAwEAEBQVhyZIlKCwsxJgxYwAA4eHhcHd3R2RkJABgypQp6N69OxYtWoR+/fohOjoaR48exRdffCHlYRAREVE9IXm4GTp0KLKysjB79mykp6ejQ4cO2LFjh27QcEpKCuTyf04wde7cGVFRUXjnnXcwc+ZMNG/eHFu2bDHoe9wQERHR/ZP8Pjd1rbbuc0NERES1pzrf37yfPhERERkUhhsiIiIyKAw3REREZFAYboiIiMigMNwQERGRQWG4ISIiIoPCcENEREQGheGGiIiIDArDDRERERkUyZdfqGu3bsisUqkkroSIiIju163v7ftZWKHBhZv8/HwAgKenp8SVEBERUXXl5+fDxsbmrm0a3NpSWq0WaWlpsLKygkwmq9HXVqlU8PT0RGpqKtetqgXs39rHPq597OPaxf6tfVL1sRAC+fn5cHNzq7SgdlUa3JkbuVwODw+PWn0Pa2tr/lLVIvZv7WMf1z72ce1i/9Y+Kfr4XmdsbuGAYiIiIjIoDDdERERkUBhuapBSqcScOXOgVCqlLsUgsX9rH/u49rGPaxf7t/bpQx83uAHFREREZNh45oaIiIgMCsMNERERGRSGGyIiIjIoDDdERERkUBhuasjy5cvh7e0NU1NTBAcH4/Dhw1KXpBciIyPRqVMnWFlZwcnJCQMHDsS5c+cqtSkpKcGkSZPQqFEjWFpa4tlnn0VGRkalNikpKejXrx/Mzc3h5OSEN954A+Xl5XV5KHpjwYIFkMlkmDp1qm4b+/jhXLt2DSNHjkSjRo1gZmaGdu3a4ejRo7rnhRCYPXs2XF1dYWZmhtDQUCQlJVV6jZycHIwYMQLW1tawtbXF2LFjUVBQUNeHUi9pNBrMmjULPj4+MDMzQ9OmTfHee+9VWmOIfVw9+/fvx4ABA+Dm5gaZTIYtW7ZUer6m+vPkyZPo2rUrTE1N4enpiQ8++KC2D013APSQoqOjhUKhEKtXrxanT58W48ePF7a2tiIjI0Pq0uq9sLAwsWbNGpGQkCDi4+NF3759hZeXlygoKNC1mThxovD09BQxMTHi6NGj4pFHHhGdO3fWPV9eXi58fX1FaGioOH78uNi+fbtwcHAQM2bMkOKQ6rXDhw8Lb29v0b59ezFlyhTddvbxg8vJyRGNGzcWo0ePFocOHRKXLl0Sv/32m7hw4YKuzYIFC4SNjY3YsmWLOHHihHjyySeFj4+PKC4u1rXp3bu38PPzE3/99Zf4448/RLNmzcSwYcOkOKR6Z968eaJRo0bi559/FpcvXxYbN24UlpaW4pNPPtG1YR9Xz/bt28Xbb78tNm3aJACIzZs3V3q+JvozLy9PODs7ixEjRoiEhASxfv16YWZmJj7//PNaPz6GmxoQFBQkJk2apPtZo9EINzc3ERkZKWFV+ikzM1MAEPv27RNCCJGbmytMTEzExo0bdW3Onj0rAIjY2FghRMUvqVwuF+np6bo2K1asENbW1qK0tLRuD6Aey8/PF82bNxc7d+4U3bt314Ub9vHDeeutt8Sjjz56x+e1Wq1wcXERH374oW5bbm6uUCqVYv369UIIIc6cOSMAiCNHjuja/Prrr0Imk4lr167VXvF6ol+/fuKFF16otO2ZZ54RI0aMEEKwjx/Wf8NNTfXnZ599Juzs7Cp9Rrz11luiZcuWtXxEQvCy1ENSq9WIi4tDaGiobptcLkdoaChiY2MlrEw/5eXlAQDs7e0BAHFxcSgrK6vUv61atYKXl5euf2NjY9GuXTs4Ozvr2oSFhUGlUuH06dN1WH39NmnSJPTr169SXwLs44e1detWBAYGYvDgwXBycoK/vz9WrVqle/7y5ctIT0+v1L82NjYIDg6u1L+2trYIDAzUtQkNDYVcLsehQ4fq7mDqqc6dOyMmJgbnz58HAJw4cQIHDhxAnz59ALCPa1pN9WdsbCy6desGhUKhaxMWFoZz587h5s2btXoMDW7hzJqWnZ0NjUZT6UMfAJydnZGYmChRVfpJq9Vi6tSp6NKlC3x9fQEA6enpUCgUsLW1rdTW2dkZ6enpujZV9f+t5wiIjo7GsWPHcOTIkdueYx8/nEuXLmHFihWIiIjAzJkzceTIEbz66qtQKBQYNWqUrn+q6r9/96+Tk1Ol542NjWFvb9/g+xcApk+fDpVKhVatWsHIyAgajQbz5s3DiBEjAIB9XMNqqj/T09Ph4+Nz22vces7Ozq5W6gcYbqgemTRpEhISEnDgwAGpSzEoqampmDJlCnbu3AlTU1OpyzE4Wq0WgYGBmD9/PgDA398fCQkJWLlyJUaNGiVxdYbh+++/x7p16xAVFYW2bdsiPj4eU6dOhZubG/uYqsTLUg/JwcEBRkZGt80sycjIgIuLi0RV6Z9XXnkFP//8M/bs2QMPDw/ddhcXF6jVauTm5lZq/+/+dXFxqbL/bz3X0MXFxSEzMxMdO3aEsbExjI2NsW/fPnz66acwNjaGs7Mz+/ghuLq6ok2bNpW2tW7dGikpKQD+6Z+7fUa4uLggMzOz0vPl5eXIyclp8P0LAG+88QamT5+O5557Du3atcPzzz+PadOmITIyEgD7uKbVVH9K+bnBcPOQFAoFAgICEBMTo9um1WoRExODkJAQCSvTD0IIvPLKK9i8eTN279592ynMgIAAmJiYVOrfc+fOISUlRde/ISEhOHXqVKVftJ07d8La2vq2L52G6IknnsCpU6cQHx+vewQGBmLEiBG6v7OPH1yXLl1uu33B+fPn0bhxYwCAj48PXFxcKvWvSqXCoUOHKvVvbm4u4uLidG12794NrVaL4ODgOjiK+q2oqAhyeeWvKyMjI2i1WgDs45pWU/0ZEhKC/fv3o6ysTNdm586daNmyZa1ekgLAqeA1ITo6WiiVSrF27Vpx5swZ8eKLLwpbW9tKM0uoai+99JKwsbERe/fuFdevX9c9ioqKdG0mTpwovLy8xO7du8XRo0dFSEiICAkJ0T1/a5pyr169RHx8vNixY4dwdHTkNOW7+PdsKSHYxw/j8OHDwtjYWMybN08kJSWJdevWCXNzc/Hdd9/p2ixYsEDY2tqKn376SZw8eVI89dRTVU6r9ff3F4cOHRIHDhwQzZs3b7DTlP9r1KhRwt3dXTcVfNOmTcLBwUG8+eabujbs4+rJz88Xx48fF8ePHxcAxOLFi8Xx48fFlStXhBA105+5ubnC2dlZPP/88yIhIUFER0cLc3NzTgXXJ0uXLhVeXl5CoVCIoKAg8ddff0ldkl4AUOVjzZo1ujbFxcXi5ZdfFnZ2dsLc3Fw8/fTT4vr165VeJzk5WfTp00eYmZkJBwcH8dprr4mysrI6Phr98d9wwz5+ONu2bRO+vr5CqVSKVq1aiS+++KLS81qtVsyaNUs4OzsLpVIpnnjiCXHu3LlKbW7cuCGGDRsmLC0thbW1tRgzZozIz8+vy8Oot1QqlZgyZYrw8vISpqamokmTJuLtt9+uNMWYfVw9e/bsqfKzd9SoUUKImuvPEydOiEcffVQolUrh7u4uFixYUCfHJxPiX7d4JCIiItJzHHNDREREBoXhhoiIiAwKww0REREZFIYbIiIiMigMN0RERGRQGG6IiIjIoDDcEBERkUFhuCEiIiKDwnBDRPXO6NGjIZPJIJPJYGJiAmdnZ/Ts2ROrV6/WrSd0P9auXQtbW9vaK5SI6iWGGyKql3r37o3r168jOTkZv/76K3r06IEpU6agf//+KC8vl7o8IqrHGG6IqF5SKpVwcXGBu7s7OnbsiJkzZ+Knn37Cr7/+irVr1wIAFi9ejHbt2sHCwgKenp54+eWXUVBQAADYu3cvxowZg7y8PN1ZoHfffRcAUFpaitdffx3u7u6wsLBAcHAw9u7dK82BElGNY7ghIr3x+OOPw8/PD5s2bQIAyOVyfPrppzh9+jS+/vpr7N69G2+++SYAoHPnzliyZAmsra1x/fp1XL9+Ha+//joA4JVXXkFsbCyio6Nx8uRJDB48GL1790ZSUpJkx0ZENYcLZxJRvTN69Gjk5uZiy5Yttz333HPP4eTJkzhz5sxtz/3www+YOHEisrOzAVSMuZk6dSpyc3N1bVJSUtCkSROkpKTAzc1Ntz00NBRBQUGYP39+jR8PEdUtY6kLICKqDiEEZDIZAGDXrl2IjIxEYmIiVCoVysvLUVJSgqKiIpibm1e5/6lTp6DRaNCiRYtK20tLS9GoUaNar5+Iah/DDRHplbNnz8LHxwfJycno378/XnrpJcybNw/29vY4cOAAxo4dC7VafcdwU1BQACMjI8TFxcHIyKjSc5aWlnVxCERUyxhuiEhv7N69G6dOncK0adMQFxcHrVaLRYsWQS6vGD74/fffV2qvUCig0WgqbfP394dGo0FmZia6du1aZ7UTUd1huCGieqm0tBTp6enQaDTIyMjAjh07EBkZif79+yM8PBwJCQkoKyvD0qVLMWDAAPz5559YuXJlpdfw9vZGQUEBYmJi4OfnB3Nzc7Ro0QIjRoxAeHg4Fi1aBH9/f2RlZSEmJgbt27dHv379JDpiIqopnC1FRPXSjh074OrqCm9vb/Tu3Rt79uzBp59+ip9++glGRkbw8/PD4sWLsXDhQvj6+mLdunWIjIys9BqdO3fGxIkTMXToUDg6OuKDDz4AAKxZswbh4eF47bXX0LJlSwwcOBBHjhyBl5eXFIdKRDWMs6WIiIjIoPDMDRERERkUhhsiIiIyKAw3REREZFAYboiIiMigMNwQERGRQWG4ISIiIoPCcENEREQGheGGiIiIDArDDRERERkUhhsiIiIyKAw3REREZFAYboiIiMig/B/4tdxPmsymxAAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential_7\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " lstm_17 (LSTM)              (None, 250, 50)           10400     \n",
            "                                                                 \n",
            " dropout_17 (Dropout)        (None, 250, 50)           0         \n",
            "                                                                 \n",
            " lstm_18 (LSTM)              (None, 50)                20200     \n",
            "                                                                 \n",
            " dropout_18 (Dropout)        (None, 50)                0         \n",
            "                                                                 \n",
            " dense_11 (Dense)            (None, 64)                3264      \n",
            "                                                                 \n",
            " dense_12 (Dense)            (None, 20)                1300      \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 35,164\n",
            "Trainable params: 35,164\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n",
            "Epoch 1/12\n",
            "4/4 [==============================] - 7s 755ms/step - loss: 0.0703 - mean_absolute_error: 0.2366 - val_loss: 0.8027 - val_mean_absolute_error: 0.8933\n",
            "Epoch 2/12\n",
            "4/4 [==============================] - 3s 717ms/step - loss: 0.0567 - mean_absolute_error: 0.2066 - val_loss: 0.6338 - val_mean_absolute_error: 0.7845\n",
            "Epoch 3/12\n",
            "4/4 [==============================] - 2s 437ms/step - loss: 0.0388 - mean_absolute_error: 0.1584 - val_loss: 0.3808 - val_mean_absolute_error: 0.5683\n",
            "Epoch 4/12\n",
            "4/4 [==============================] - 2s 439ms/step - loss: 0.0255 - mean_absolute_error: 0.1240 - val_loss: 0.2526 - val_mean_absolute_error: 0.4574\n",
            "Epoch 5/12\n",
            "4/4 [==============================] - 2s 415ms/step - loss: 0.0160 - mean_absolute_error: 0.0972 - val_loss: 0.2600 - val_mean_absolute_error: 0.4938\n",
            "Epoch 6/12\n",
            "4/4 [==============================] - 2s 419ms/step - loss: 0.0111 - mean_absolute_error: 0.0798 - val_loss: 0.1843 - val_mean_absolute_error: 0.4157\n",
            "Epoch 7/12\n",
            "2/4 [==============>...............] - ETA: 0s - loss: 0.0085 - mean_absolute_error: 0.0701"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 2s 2s/step\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjcAAAHHCAYAAABDUnkqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABR3UlEQVR4nO3deXhM1/8H8PdMlsm+kD1CYt8iIpFYqqqNnVYXlJRQ2mrRkG60Rf20wrdoWpRqi2rtbdEqitiK1JIIgkRiC5EVyWSRSWbm/P5ITQ2hCUluZvJ+Pc88ydx77p3PvUje7j33HJkQQoCIiIjISMilLoCIiIioKjHcEBERkVFhuCEiIiKjwnBDRERERoXhhoiIiIwKww0REREZFYYbIiIiMioMN0RERGRUGG6IiIjIqDDcEFG18Pb2xqhRo3Tv9+3bB5lMhn379klW073urZGIjAPDDZERWrlyJWQyme5lYWGB5s2bY8KECcjMzJS6vErZtm0bPvnkE0lrkMlkmDBhwkPbaLVarFq1CsHBwahXrx5sbW3RvHlzjBw5En///TeAsjB195/Lg14rV67Ufa5MJsPYsWPL/cyPPvpI1yYnJ6dKj5nIkJlKXQARVZ//+7//g4+PD4qLi3Hw4EEsWbIE27ZtQ0JCAqysrGq0lieffBK3b9+Gubl5pbbbtm0bFi9eLHnA+S9vv/02Fi9ejOeeew6hoaEwNTVFUlIStm/fjsaNG6NTp06IiopCQUGBbptt27Zh7dq1+OKLL+Dk5KRb3qVLF933FhYW+OWXX/D111/fd+7Wrl0LCwsLFBcXV/8BEhkQhhsiI9a3b18EBgYCAMaOHYv69etjwYIF2LJlC4YNG1buNoWFhbC2tq7yWuRyOSwsLKp8v7VBZmYmvv76a7z22mtYtmyZ3rqoqChkZ2cDAAYNGqS3LiMjA2vXrsWgQYPg7e1d7r779OmD3377Ddu3b8dzzz2nW3748GFcunQJL774In755ZcqPR4iQ8fbUkR1yNNPPw0AuHTpEgBg1KhRsLGxwYULF9CvXz/Y2toiNDQUQNltlqioKLRp0wYWFhZwdXXFG2+8gVu3buntUwiBTz/9FA0aNICVlRV69OiBM2fO3PfZD+pzc+TIEfTr1w+Ojo6wtrZGu3bt8OWXX+rqW7x4MQDo3ba5o6prfFSXLl2CEAJdu3a9b51MJoOLi8sj79vT0xNPPvkk1qxZo7d89erV8PX1Rdu2bR9530TGilduiOqQCxcuAADq16+vW6ZWq9G7d2888cQTmDdvnu521RtvvIGVK1di9OjRePvtt3Hp0iUsWrQIJ06cwKFDh2BmZgYAmD59Oj799FP069cP/fr1Q1xcHHr16oWSkpL/rGfXrl0YMGAA3N3dER4eDjc3N5w7dw5bt25FeHg43njjDVy/fh27du3Cjz/+eN/2NVFjRTRq1AgAsHHjRgwePLjKb/kNHz4c4eHhKCgogI2NDdRqNTZu3IiIiAjekiIqjyAio7NixQoBQOzevVtkZ2eLq1evinXr1on69esLS0tLce3aNSGEEGFhYQKAmDJlit72f/31lwAgVq9erbd8x44desuzsrKEubm56N+/v9Bqtbp2H374oQAgwsLCdMv27t0rAIi9e/cKIYRQq9XCx8dHNGrUSNy6dUvvc+7e1/jx40V5P6qqo8YHASDGjx//0DYjR44UAISjo6N4/vnnxbx588S5c+ceus3nn38uAIhLly499HNv3rwpzM3NxY8//iiEEOKPP/4QMplMXL58WcyYMUMAENnZ2f95HER1BW9LERmxkJAQODs7w8vLCy+//DJsbGywadMmeHp66rV788039d5v3LgR9vb26NmzJ3JycnSvgIAA2NjYYO/evQCA3bt3o6SkBBMnTtS7XTRp0qT/rO3EiRO4dOkSJk2aBAcHB711d+/rQWqixspYsWIFFi1aBB8fH2zatAnvvvsuWrVqhWeeeQZpaWmPtW9HR0f06dMHa9euBQCsWbMGXbp00V0xIiJ9dTrcHDhwAAMHDoSHhwdkMhk2b95c6X0IITBv3jw0b94cCoUCnp6e+Oyzz6q+WKJHsHjxYuzatQt79+7F2bNncfHiRfTu3VuvjampKRo0aKC3LDk5GXl5eXBxcYGzs7Peq6CgAFlZWQCAK1euAACaNWumt72zszMcHR0fWtudW2SP2mekJmqsDLlcjvHjxyM2NhY5OTnYsmUL+vbtiz179uDll19+7P0PHz4cu3btQmpqKjZv3ozhw4dXQdVExqlO97kpLCyEn58fXn31VbzwwguPtI/w8HDs3LkT8+bNg6+vL27evImbN29WcaVEjyYoKEj3tNSDKBQKyOX6/8/RarVwcXHB6tWry93G2dm5ymp8VLW5xvr16+PZZ5/Fs88+i6eeegr79+/HlStXHutKy7PPPguFQoGwsDCoVCoMGTKkCismMi51Otz07dsXffv2feB6lUqFjz76CGvXrkVubi7atm2LuXPn4qmnngIAnDt3DkuWLEFCQgJatGgBAPDx8amJ0omqVZMmTbB792507doVlpaWD2x355d1cnIyGjdurFuenZ193xNL5X0GACQkJCAkJOSB7R50i6omaqwKgYGB2L9/P9LT0x8r3FhaWmLQoEH46aef0LdvX71xcYhIX52+LfVfJkyYgJiYGKxbtw6nTp3C4MGD0adPHyQnJwMAfv/9dzRu3Bhbt26Fj48PvL29MXbsWF65IYM3ZMgQaDQazJo16751arUaubm5AMr69JiZmWHhwoUQQujaREVF/edndOjQAT4+PoiKitLt746793VnzJ1729REjRWVkZGBs2fP3re8pKQE0dHRkMvlaNq06WN/zrvvvosZM2Zg2rRpj70vImNWp6/cPExqaipWrFiB1NRUeHh4ACj7wbJjxw6sWLECs2fPxsWLF3HlyhVs3LgRq1atgkajweTJk/HSSy9hz549Eh8B0aPr3r073njjDURGRiI+Ph69evWCmZkZkpOTsXHjRnz55Zd46aWX4OzsjHfffReRkZEYMGAA+vXrhxMnTmD79u3/eWVBLpdjyZIlGDhwINq3b4/Ro0fD3d0diYmJOHPmDP78808AQEBAAICyEYB79+4NExMTvPzyyzVS492OHz+OTz/99L7lTz31FCwsLBAUFISnn34azzzzDNzc3JCVlYW1a9fi5MmTmDRpUpVcafHz84Ofn99j74fI6En7sFbtAUBs2rRJ937r1q0CgLC2ttZ7mZqaiiFDhgghhHjttdcEAJGUlKTbLjY2VgAQiYmJNX0IRDp3HgU/duzYQ9uFhYUJa2vrB65ftmyZCAgIEJaWlsLW1lb4+vqK999/X1y/fl3XRqPRiJkzZwp3d3dhaWkpnnrqKZGQkCAaNWr00EfB7zh48KDo2bOnsLW1FdbW1qJdu3Zi4cKFuvVqtVpMnDhRODs7C5lMdt9j4VVZ44MAeOBr1qxZQqlUii+//FL07t1bNGjQQJiZmQlbW1vRuXNn8e233+o9gn63ij4K/jB8FJzofjIh7rpOW4fJZDJs2rRJNzz6+vXrERoaijNnzsDExESvrY2NDdzc3DBjxgzMnj0bpaWlunW3b9+GlZUVdu7ciZ49e9bkIRARERF4W+qB/P39odFokJWVhW7dupXbpmvXrlCr1bhw4YKuc+T58+cBgONPEBERSaROX7kpKChASkoKgLIws2DBAvTo0QP16tVDw4YN8corr+DQoUOYP38+/P39kZ2djejoaLRr1w79+/eHVqtFx44dYWNjg6ioKGi1WowfPx52dnbYuXOnxEdHRERUN9XpcLNv3z706NHjvuVhYWFYuXIlSktL8emnn2LVqlVIS0uDk5MTOnXqhJkzZ8LX1xcAcP36dUycOBE7d+6EtbU1+vbti/nz56NevXo1fThERESEOh5uiIiIyPhwnBsiIiIyKgw3REREZFQkfVrqwIED+PzzzxEbG4v09HS9R7HL8+uvv2LJkiWIj4+HSqVCmzZt8Mknn9w3EeDDaLVaXL9+Hba2thWaeZiIiIikJ4RAfn4+PDw87psP716ShpvKTlx54MAB9OzZE7Nnz4aDgwNWrFiBgQMH4siRI/D396/QZ16/fh1eXl6PWzoRERFJ4OrVq2jQoMFD29SaDsX3DqJXUW3atMHQoUMxffr0CrXPy8uDg4MDrl69Cjs7u0eolIiIiGqaUqmEl5cXcnNzYW9v/9C2Bj2In1arRX5+fqUeu75zK8rOzo7hhoiIyMBUpEuJQYebefPmoaCgAEOGDHlgG5VKBZVKpXuvVCprojQiIiKSiME+LbVmzRrMnDkTGzZsgIuLywPbRUZGwt7eXvdifxsiIiLjZpDhZt26dRg7diw2bNiAkJCQh7adOnUq8vLydK+rV6/WUJVEREQkBYO7LbV27Vq8+uqrWLduHfr37/+f7RUKBRQKRQ1URkRERLWBpOHm7okrAeDSpUuIj4/XTVw5depUpKWlYdWqVQDKbkWFhYXhyy+/RHBwMDIyMgAAlpaW/9lzmoiIiOoGSW9LHT9+HP7+/roxaiIiIuDv7697rDs9PR2pqam69suWLYNarcb48ePh7u6ue4WHh0tSPxEREdU+tWacm5qiVCphb2+PvLw8PgpORERkICrz+9sgOxQTERERPQjDDRERERkVhhsiIiIyKgw3REREZFQYboiIiMioMNwQERHRoxMCuH1b6ir0MNwQERHRo3v2WcDKCrh+XepKdBhuiIiI6NFt3Vr29aefpK3jLgw3RERE9PhMTKSuQIfhhoiIiB6fvPZEitpTCRERERkuXrkhIiIio8JwQ0REREaF4YaIiIiMCvvcEBERkcHTaP79nuGGiIiIDF5p6b/f87YUERERGaS4OOCbb8qmXbg73NSiKzemUhdAREREBqJfP2D79rLvnZ2B7t3/XccrN0RERGRwhPj3+4QE/Ss3tQjDDREREVWM6V03fGQy/XBzd+diiTHcEBERUcXcHW7kcqCk5N/3Wm3N1/MADDdERERUMWZm/36vUvHKDRERERm4u6/czJoF3Lr173uGGyIiIjI4pvc8ZL1//7/fM9wQERGRwbk33LDPDRERERm0h4UbXrkhIiIig/OwcLN/P3D8eM3W8wAMN0RERFQxDws3mzYBHTvWittTDDdERERUMffOH1VcfH8btbpmankIhhsiIiKqGJlM/31+/v1tasGUDAw3RERE9GjKCzd336qSCMMNERERPRql8v5lDDdERERksO4exO8O3pYiIiIio8IrN0RERGQw7u1QXB6GGyIiIjIqvC1FRERERoVXboiIiMio8MoNERERGRVeuSEiIiKjwnBDRERERoW3pYiIiMionDsndQUMN0RERFSF3nkH2L1b0hIYboiIiKhqvf22pB/PcENERERVKygIUKsl+3hTyT6ZiIiIjNPKlZJ+PK/cEBERUcUMGSJ1BRXCcENEREQV06kT4Oz88DZdu9ZMLQ8habg5cOAABg4cCA8PD8hkMmzevPk/t9m3bx86dOgAhUKBpk2bYqXEl76IiIjqlObNH75+/fqaqeMhJA03hYWF8PPzw+LFiyvU/tKlS+jfvz969OiB+Ph4TJo0CWPHjsWff/5ZzZUSERERAEAI/fceHvrvPT1rrpYHkLRDcd++fdG3b98Kt1+6dCl8fHwwf/58AECrVq1w8OBBfPHFF+jdu3d1lUlEREQAbpdoYHnvwnvDTi1gUH1uYmJiEBISoresd+/eiImJeeA2KpUKSqVS70VERESVU1yqwYtLDiP1ZpH+Coabx5ORkQFXV1e9Za6urlAqlbh9+3a520RGRsLe3l738vLyqolSiYiIjMonv53B2XQlbhbeMzEmw03Nmzp1KvLy8nSvq1evSl0SERGRQfk59hrWHbsKmQxo7GStv7IWhhuDGsTPzc0NmZmZessyMzNhZ2cHS8v77gICABQKBRQKRU2UR0REZHQSM5T4ePNpAMDkkOaw239PdKiF4cagrtx0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\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from statsmodels.tsa.api import SARIMAX\n",
        "from statsmodels.tsa.stattools import adfuller\n",
        "from statsmodels.tsa.arima.model import ARIMA\n",
        "import numpy as np # Load the data i\n",
        "\n",
        "\n",
        "import math\n",
        "def tst(num=1000):\n",
        "    arr=[]\n",
        "    for i in range(0,num):\n",
        "      arr.append(i*i+i)\n",
        "    return arr\n",
        "\n",
        "\n",
        "n_lookback=250\n",
        "n_forecast=20\n",
        "def create_sequence(dataset ):\n",
        "     X = []\n",
        "     Y = []\n",
        "     for i in range(n_lookback, len(dataset) - n_forecast + 1):\n",
        "        X.append(dataset[i - n_lookback: i])\n",
        "        Y.append(dataset[i: i + n_forecast])\n",
        "\n",
        "     X = np.array(X)\n",
        "     Y = np.array(Y)\n",
        "\n",
        "     return(X,Y)\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "dtmp=tst()\n",
        "\n",
        "\n",
        "df=pd.DataFrame(data=dtmp,columns=['Value'])\n",
        "df['Date']=range(0,len(dtmp))\n",
        "df=df[['Date','Value']]\n",
        "df=df.set_index('Date')\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "#df = yf.download(coin_,  start='2022-01-01', end=dt.now() , interval=interval_) #data = pd.read_csv('data.csv', index_col='Date', parse_dates=True)\n",
        "\n",
        "# Set the frequency of the index to monthly #\n",
        "#data = data.resample('M').asfreq()\n",
        "data=df\n",
        "data.dropna().astype (float)\n",
        "\n",
        "# Plot the data\n",
        "data.plot()\n",
        "plt.title('Time Series Data')\n",
        "plt.xlabel('Date')\n",
        "plt.ylabel('Value')\n",
        "plt.show()\n",
        "\n",
        "# Check the stationarity of the time series\n",
        "result = adfuller(data['Value'])\n",
        "print(f'ADF Statistic: {result[0]}')\n",
        "print(f'p-value: {result[1]}')\n",
        "for key, value in result[4].items():\n",
        "    print('Critical Values:')\n",
        "    print(f'   {key}: {value}')\n",
        "\n",
        "# Apply first-order differencing to make the time series stationary\n",
        "diffs = data.diff().dropna()\n",
        "\n",
        "# Plot the differenced data\n",
        "diffs.plot()\n",
        "plt.title('Differenced Time Series Data')\n",
        "plt.xlabel('Date')\n",
        "plt.ylabel('Value')\n",
        "plt.show()\n",
        "##-----------------------------------------------------------------------------------------\n",
        "# Check the stationarity of the differenced time series\n",
        "result = adfuller(diffs['Value'])\n",
        "print(f'ADF Statistic: {result[0]}')\n",
        "print(f'p-value: {result[1]}')\n",
        "for key, value in result[4].items():\n",
        "    print('Critical Values:')\n",
        "    print(f'   {key}: {value}')\n",
        "\n",
        "# Define the SARIMAX model\n",
        "model = SARIMAX(endog=diffs, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))\n",
        "\n",
        "# Fit the SARIMAX model\n",
        "results = model.fit()\n",
        "\n",
        "# Print the model summary\n",
        "print(results.summary())\n",
        "\n",
        "# Make predictions for the next 100 periods\n",
        "forecast = results.forecast(steps=n_forecast)\n",
        "\n",
        "# Convert the differenced forecast back to the original scale\n",
        "last_value = data.iloc[-1]['Value']\n",
        "forecast = forecast.cumsum() + last_value\n",
        "\n",
        "# Plot the predicted values\n",
        "plt.plot(df['Value'])\n",
        "plt.plot(forecast, color='red')\n",
        "plt.title('Predicted Values')\n",
        "plt.xlabel('Date')\n",
        "plt.ylabel('Value')\n",
        "plt.show()\n",
        "\n",
        "#-------------------------LSTM\n",
        "\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "\n",
        "scaler = MinMaxScaler(feature_range=(0, 1))\n",
        "scaler=scaler.fit(df)\n",
        "dft=scaler.transform(df)\n",
        "\n",
        "training_size = round(len(dft) * 0.7)\n",
        "train_data = dft[:training_size]\n",
        "test_data = dft[training_size:]\n",
        "\n",
        "X,Y=create_sequence(train_data)\n",
        "Xtest,Ytest=create_sequence(test_data)\n",
        "from keras.models import Sequential,load_model\n",
        "from keras.layers import Dense, Dropout, LSTM, Bidirectional\n",
        "\n",
        "\n",
        "model = Sequential()\n",
        "model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))\n",
        "model.add(Dropout(0.4))\n",
        "model.add(LSTM(units=50))\n",
        "model.add(Dropout(0.4))\n",
        "model.add(Dense(64, activation='relu'))\n",
        "model.add(Dense(n_forecast))\n",
        "\n",
        "model.compile(loss='mean_squared_error', metrics=['mean_absolute_error'], optimizer='adam')\n",
        "model.summary()\n",
        "history=model.fit(X, Y, epochs=12, batch_size=128, verbose=1, validation_data=(Xtest,Ytest))\n",
        "\n",
        "plt.plot(history.history['loss'], 'r', label='Training loss')\n",
        "plt.plot(history.history['val_loss'], 'g', label='Validation loss')\n",
        "plt.title('Training VS Validation loss')\n",
        "plt.xlabel('No. of Epochs')\n",
        "plt.ylabel('Loss')\n",
        "plt.legend()\n",
        "plt.show()\n",
        "\n",
        "\n",
        "X_ = dft[- n_lookback:].reshape(1,-1)\n",
        "\n",
        "Y_=model.predict(X_)\n",
        "\n",
        "YP=scaler.inverse_transform(Y_)\n",
        "\n",
        "YP=pd.DataFrame(data=YP.T,columns=[\"Value\"])\n",
        "YP['Date']=range(len(dtmp),len(dtmp)+n_forecast)\n",
        "YP=YP.set_index('Date')\n",
        "# Convert the differenced forecast back to the original scale\n",
        "\n",
        "# Plot the predicted values\n",
        "plt.plot(df['Value'])\n",
        "plt.plot(YP, color='red')\n",
        "plt.title('Predicted LSTM')\n",
        "plt.xlabel('Date')\n",
        "plt.ylabel('Value')\n",
        "plt.show()\n",
        "\n",
        "\n"
      ]
    }
  ]
}


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np # Load the data i


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
model.add(Dense(64, activation='relu'))
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







