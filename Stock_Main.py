
#IMPORTING ALL NECESSARY PACKAGES

import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')




st.title("DATA VISUALISATION AND PREDICTION OF STOCKS")



# DECLARING THE TIME PERIOD FOR THE DATA WE ARE GOING TO USE

st.header("Declare the time period for the data we are going to use")

START=st.date_input("Date")
TODAY = date.today().strftime("%Y-%m-%d")




# THE STOCKS WE ARE GOING TO USE

STOCK = st.text_input("Enter Stock Name")




# DECLARING A FUNCTION TO GET AND LOAD THE DATA


def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(STOCK)


# VIEWING THE LOADED DATA

df = pd.DataFrame(data)

st.header("Dataset")
st.dataframe(df)




# DETERMINING THE SHAPE OF THE DATA

st.header("Shape of the data")
data_shape=data.shape
st.subheader(data_shape)


# CHECKING WHETHER THE DATA HAS ANY NULL VALUES

null_in_data = data.isnull().sum()
st.header("Any null values ")
st.dataframe(null_in_data)


# CHECKING THE DATATYPE OF EACH COLUMN

data_types = data.dtypes 
st.header("Data types of each column")
data_types_df = pd.DataFrame(data_types)
st.text(data_types)


# DESCRIBING THE DATA

described_data = data.describe().astype(int)
st.header("Describing the dataset")
st.dataframe(described_data)


# PLOTTING TIME SERIES DATA

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data ['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data ['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text = 'Time Series Data',xaxis_rangeslider_visible = True )
    fig.write_image("Time_Series_Data.png")
    st.plotly_chart(fig, use_container_width=True)



st.header("Plotting time series data")
plot_raw_data()

# CHECKING THE DAILY RETURNS

data['Day_Perc_Change'] = data['Adj Close'].pct_change()*100
data.dropna(axis = 0, inplace = True)

daily_percent_change_df = pd.DataFrame(data)

st.header("Checking the Daily Percent Change")
st.dataframe(daily_percent_change_df)


# PLOTTING DAILY RETURNS

def plot_dr_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data ['Date'], y=data['Day_Perc_Change']))
    fig.layout.update(title_text = 'Daily Returns',xaxis_rangeslider_visible = True )
    fig.write_image("Daily_Returns.png")
    st.plotly_chart(fig, use_container_width=True)


st.header("Plotting daily returns")
plot_dr_data()


# DESCRIBING DAY PERCENT CHANGE

described_day_percent_change = data.Day_Perc_Change.describe().astype(int)
st.header("Describing the daily percent change")
st.dataframe(described_data)

# PLOTTING DAILY RETURNS HISTOGRAM DISTRIBUTION

def plot_drhd_data():
    fig = px.histogram(data['Day_Perc_Change'])
    fig.write_image("Daily_Returns_HD.png")
    st.plotly_chart(fig, use_container_width=True)


st.header("Plotting Daily Returns histogram distribution")
plot_drhd_data()

# TREND ANALYSIS OF THE GIVEN DATA

def trend(x):
  if x > -0.5 and x <= 0.5:
    return 'Slight or No change'
  elif x > 0.5 and x <= 1:
    return 'Slight Positive'
  elif x > -1 and x <= -0.5:
    return 'Slight Negative'
  elif x > 1 and x <= 3:
    return 'Positive'
  elif x > -3 and x <= -1:
    return 'Negative'
  elif x > 3 and x <= 7:
    return 'Among top gainers'
  elif x > -7 and x <= -3:
    return 'Among top losers'
  elif x > 7:
    return 'Bull run'
  elif x <= -7:
    return 'Bear drop'
data['Trend']= np.zeros(data['Day_Perc_Change'].count())
data['Trend']= data['Day_Perc_Change'].apply(lambda x:trend(x))
st.header("Trend Analysis of the data")
st.dataframe(data)



# VISUALIZING DAILY RETURNS TREND

def plot_drt_data():
    Trend_Value=data["Trend"]
    Day_Perc_Change_Value = data['Day_Perc_Change']
    fig = px.pie(data,names=Trend_Value,values=Day_Perc_Change_Value)

    fig = px.pie(data, values='Day_Perc_Change', names='Trend')

    fig.update_traces(textinfo="percent + value")
    fig.write_image("Daily_Returns_Trend.png")
    st.plotly_chart(fig, use_container_width=True)


st.header("Visualizing daily returns trend")
plot_drt_data()


# VISUALIZING VOLATALITY OF THE STOCK

def plot_dv_data():
    data_vol = data['Adj Close'].rolling(7).std()*np.sqrt(7)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data ['Date'], y=data_vol))
    fig.layout.update(title_text = 'Volatility Analysis',xaxis_rangeslider_visible = True )
    fig.write_image("Volatality_Analysis.png")
    st.plotly_chart(fig, use_container_width=True)


st.header("Visualizing volatality of the stock")
plot_dv_data()


# PREDICTING AND VISUALISING THE PREDICTED VALUE

df_train = data[['Date','Close']]
df_train = df_train. rename(columns={"Date": "ds", "Close": "y"})
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

fig1 = plot_plotly(m, forecast)
fig1.write_image("Forecasted_Trend.png")
fig2 = m.plot_components(forecast)

st.header("Predicting and visualising the predicted value")
st.plotly_chart(fig1, use_container_width=True)







