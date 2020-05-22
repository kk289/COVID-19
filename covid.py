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

  


