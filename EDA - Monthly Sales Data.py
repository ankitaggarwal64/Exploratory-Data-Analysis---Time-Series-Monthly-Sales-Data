# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
## ARIMA Modelling

# De
fining a Validation Dataset.

import pandas as pd
totalseries = pd.read_csv('D:\Data Science Projects\Time series forecasting\Arima Implementation\ABC4wd.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
#header=0: We must specify the header information at row 0.
#parse dates=True: We give the function a hint that data in the fi
rst column contains dates that need to be parsed.
#index col=0: We hint that the fi
rst column contains the index information for the timeseries.
#squeeze=True: We hint that we only have one data column and that we are interested in a Series and not a DataFrame.
--------------------------------
totalseries1 = pd.read_csv('D:\Data Science Projects\Time series forecasting\Arima Implementation\ABC4wd.csv')
print(totalseries1.dtypes)
dateparse = lambda Date: pd.datetime.strptime(Date, '%b-%y')
data = pd.read_csv('D:\Data Science Projects\Time series forecasting\Arima Implementation\ABC4wd.csv', parse_dates=['Date'], index_col='Date',date_parser=dateparse)
print ('\n Parsed Data:')
print (data.head())

series1= data.iloc[:,0]
split_point = len(series1)-12

dataset1,validation1 = series1[0:split_point],series1[split_point:]
print('Dataset %d, validation %d' % (len(dataset1),len(validation1)))
dataset1.to_csv('D:\Data Science Projects\Time series forecasting\Arima Implementation\dataset1.csv',header =False)
validation1.to_csv('D:\Data Science Projects\Time series forecasting\Arima Implementation\\validation1.csv',header =False)
---------------------------------

series = totalseries.iloc[:,0]
split_point = len(series)-12

dataset,validation = series[0:split_point],series[split_point:]
print('Dataset %d, validation %d' % (len(dataset),len(validation)))

dataset.to_csv('D:\Data Science Projects\Time series forecasting\Arima Implementation\dataset.csv',header =False)
validation.to_csv('D:\Data Science Projects\Time series forecasting\Arima Implementation\\validation.csv',header =False)

# Developing a Method for Model Evaluation.

#Persistence Model (Baseline performance - naive forecast)
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
#load data
dataset = pd.read_csv('D:\Data Science Projects\Time series forecasting\Arima Implementation\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
X = dataset.values
X=X.astype('float32')
train_size = int(len(X)*0.5)
train,test = X[0:train_size], X[train_size:]

# walk forward validation
history = [x for x in train]

predictions = list()
for i in range(len(test)):
    yhat = history[-1]
    predictions.append(yhat)
    obs = test[i]
    history.append(obs)
    print('Prediction %d, Expected %d' %(yhat,obs))

# print performance
rmse = sqrt(mean_squared_error(test,predictions))
print('RMSE: %.3f' % rmse)

forecast_errors = [test[i]-predictions[i] for i in range(len(test))]
bias = sum(forecast_errors)*1.0/len(test)
print('Bias: %.3f' % bias)

percent_forecast_errors = [abs(test[i]-predictions[i])*100/test[i] for i in range(len(test))]
MAPE = sum(percent_forecast_errors)*1.0/len(test)
print('MAPE: %.3f' % MAPE)


## DATA ANALYSIS

# summmary statistics
import pandas as pd
dataset = pd.read_csv('D:\Data Science Projects\Time series forecasting\Arima Implementation\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
print(dataset.describe())

# line plot of time series
from matplotlib import pyplot
dataset = pd.read_csv('D:\Data Science Projects\Time series forecasting\Arima Implementation\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
pyplot.figure(figsize=(12,10))
dataset.plot()
pyplot.show()
## Obseravtion : There is clear cyclicity and seasonality

# seasonal plots
# multiple line plots of time series
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot
dataset1 = read_csv('D:\Data Science Projects\Time series forecasting\Arima Implementation\dataset1.csv',index_col=0,header=None, parse_dates=True, squeeze=True)

groups = dataset1['2002':'2015'].groupby(Grouper(freq='A'))
groups.head()

years = DataFrame()
pyplot.figure(figsize=(12,10))
n_groups = len(groups)
i = 1
for name, group in groups:
    pyplot.subplot(n_groups,1, i)
    i += 1
    pyplot.plot(group)
pyplot.show()

## Obseravtion : Clear seasonality , highest sales in march and august


# density plots of time series
from pandas import read_csv
from matplotlib import pyplot
dataset1 = read_csv('D:\Data Science Projects\Time series forecasting\Arima Implementation\dataset1.csv',index_col=0,header=None, parse_dates=True, squeeze=True)
pyplot.figure(1)
pyplot.subplot(211)
series.hist()
pyplot.subplot(212)
series.plot(kind='kde')
pyplot.show()

# Obseravtion : The distribution is not Gaussian.
# Obseravtion :The shape has a long right tail and may suggest an exponential distribution. Power transform can be helpful

# boxplots of time series
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot
dataset1 = read_csv('D:\Data Science Projects\Time series forecasting\Arima Implementation\dataset1.csv',index_col=0,header=None, parse_dates=True, squeeze=True)
groups = dataset1['1990':'2015'].groupby(Grouper(freq='A'))
years = DataFrame()
pyplot.figure(figsize=(12,10))
for name, group in groups:
    years[name.year] = group.values
    years.boxplot()
pyplot.show()

##Obseravtion : definitely there seems to be cyclicity and outliers seems to be part of seasonal cycle
