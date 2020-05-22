import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from fbprophet import Prophet
import seaborn as sns
# %matplotlib inline

#Load the csv data, creating pandas dataframe by persisting date format
world_df = pd.read_csv('https://raw.githubusercontent.com/bitprj/covid19/master/covid_data.csv',parse_dates=['Date'])

#Display top 5 records
print(world_df.head(10))

#view some basic statistical details like percentile, mean, std 
world_df.describe()

world_df.info()

#whether the data frame has any NAN values
world_df.isnull().values.any()

# Preprocessing the dataset
#Replace NAN values with blank string
world_df.fillna({'Province_State': ""},inplace=True)

#Drop unusefull columns if any 
world_df.drop("Id",axis=1,inplace=True)

# Data Visualization
#bar chart for total confirmed and fatalities cases
axis = world_df[['Date','ConfirmedCases','Fatalities']].set_index('Date').plot(figsize=(12, 8),logy=True)
plt.show()

# Data Aggregation
world_df.groupby('Country_Region')

country_cases_df = world_df.groupby('Country_Region')['ConfirmedCases'].sum().reset_index()
country_cases_df.sort_values(["ConfirmedCases"],ascending=False,inplace=True)

# 10 hotspots of covid 19 cases
top_count = 10
country_cases_df[:top_count].plot.bar(x = 'Country_Region', y ='ConfirmedCases')
plt.show()

# most death rated countries

#Find sum of deaths per country
country_death_df = world_df.groupby('Country_Region')['Fatalities'].sum().reset_index()

#merge grouped cases and grouped deaths
total_df = country_death_df.merge(country_cases_df)


#Find death rate and sort in descending order
total_df['DeathRate']   = (total_df.Fatalities / total_df.ConfirmedCases)*100
total_df.sort_values('DeathRate',ascending=False,inplace=True)

#bar chart with x = death rate and y = countries
axis = total_df.head(5).plot.barh(x='Country_Region',y = 'DeathRate');
plt.show()

# Train and test split
def train_test_data_split(df,period=5):
  #Excluding last 5 for the training set
  train_data = df[:-period]
  #Including last 5 for the test set
  test_data =  df[-period:]
  return train_data,test_data

# Prophet Model Training
def modeling(train_data):
  # Prophet algorith helps to predict the time series data
  pm = Prophet(changepoint_prior_scale=0.95,interval_width=1)
  #training the model
  pm.fit(train_data)
  return pm

 # Predictions
def predictions(pm,periods=5):
  #considering 5 future days
  future = pm.make_future_dataframe(periods)
  #predicting provided days
  pm_forecast = pm.predict(future)
  return pm_forecast

# Model Evaluation
def mean_absolute_percentage_error(y_true, y_pred):
   y_true, y_pred = np.array(y_true), np.array(y_pred)
   return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Helper Function
# Main function which helps to run the data preparation,model execution, predictions and evaluations
def helper(df,countries):
  # train/test split, fitting the model, and predicting (5) days ahead
  train_data,test_data = train_test_data_split(df,5)
  pm = modeling(train_data)
  predictions_df = predictions(pm,5)
 
  #finalizes resulting data frame output
  results_df = predictions_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(test_data)
  print(results_df)

  #plots the historical data and predictions
  fig1 = pm.plot(predictions_df)
  fig1.suptitle(country,verticalalignment='center')
  fig2 = pm.plot_components(predictions_df)
  fig2.suptitle(country,verticalalignment='center')
  print(country+' MAPE: '+str(mean_absolute_percentage_error(results_df['yhat'].iloc[0], results_df['y'].iloc[0])))
  return results_df

# Predict future cases for a given country?
country_df = world_df.groupby(['Country_Region','Date'])[['ConfirmedCases','Fatalities']].sum().reset_index()

def countryWiseCasesPredictions(df,country):
  
  #grouping country wise
  country_df = world_df.groupby(['Country_Region','Date'])[['ConfirmedCases','Fatalities']].sum().reset_index()
  #filtering provided country data
  country_df = country_df[country_df.Country_Region == country]
  #Renaming column to fit prophet's model 
  country_rn_df = country_df.rename(columns={"Date":"ds","ConfirmedCases":"y"})
  #Dropping unused columns
  country_rn_df = country_rn_df.reset_index().drop(["index","Fatalities","Country_Region"],axis= 1)
  return country_rn_df

#Comma sepearted list for predictions
# countries = ['China','Thailand','Canada']
countries = ['Canada']
for country in countries:
   df = countryWiseCasesPredictions(world_df,country)
   results = helper(df,countries)
   results.head()

# Predict total deaths for a given country

def countryWiseDeathPredictions(df,country):
  #grouping country wise
  q3_df = df.groupby(['Country_Region','Date'])[['ConfirmedCases','Fatalities']].sum().reset_index()
  #filtering provided country data
  q3_df = q3_df[q3_df.Country_Region == country]
  #Renaming column as per the models needs
  q3_rn_df = q3_df.rename(columns={"Date":"ds","Fatalities":"y"})
  #Dropping unused columns
  q3_rn_df = q3_rn_df.reset_index().drop(["index","ConfirmedCases","Country_Region"],axis= 1)
  return q3_rn_df


countries = ['Italy']
for country in countries:
   df = countryWiseDeathPredictions(world_df,country)
   results = helper(df,countries)
   results.head()

