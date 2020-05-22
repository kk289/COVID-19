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
