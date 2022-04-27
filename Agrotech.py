#!/usr/bin/env python
# coding: utf-8

# # Import Required Libraries

# In[1]:


import matplotlib
import pandas as pd
import seaborn as sns
import numpy as np

from dateutil import parser

from numpy import absolute
from numpy import mean
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import itertools
import sklearn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor, BayesianRidge
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.dummy import DummyRegressor


# # Read Data

# In[2]:


df = pd.ExcelFile('Data.xlsx')


# In[3]:


plants = pd.read_excel(df, 'plants')

plants  = plants.rename(columns = {'Batch Number': 'batch_number', 
                                   'Plant Date': 'plant_date', 
                                   'Class': 'class',
                                   'Fresh Weight (g)': 'fresh_weight', 
                                   'Head Weight (g)': 'head_weight',
                                   'Radial Diameter (mm)': 'radial_diameter', 
                                   'Polar Diameter (mm)': 'polar_diameter',
                                   'Diameter Ratio': 'diameter_ratio', 
                                   'Leaves': 'leaves',
                                   'Density (kg/L)': 'density',
                                   'Leaf Area (cm^2)': 'leaf_area',
                                   'Square ID': 'square_id',
                                   'Check Date': 'check_date', 
                                   'Flight Date': 'flight_date',
                                   'Remove': 'remove'})

print(plants)


# In[4]:


planting = pd.read_excel(df, 'planting')

print(planting)


# In[5]:


weather = pd.read_excel(df, 'weather')

weather  = weather.rename(columns = {'' :'weather_date',
                                     'Solar Radiation [avg]' : 'solar_radiation',
                                     'Precipitation [sum]' : 'precipitation',
                                     'Wind Speed [avg]' : 'wind_speed_avg',
                                     'Wind Speed [max]' : 'wind_speed_max',
                                     'Battery Voltage [last]' : 'battery_voltage',
                                     'Leaf Wetness [time]' : 'leaf_wetness',
                                     'Air Temperature [avg]' : 'air_temp_avg',
                                     'Air Temperature [max]' : 'air_temp_max',
                                     'Air Temperature [min]' : 'air_temp_min',
                                     'Relative Humidity [avg]' : 'relative_humidity',
                                     'Dew Point [avg]' : 'dew_point_avg',
                                     'Dew Point [min]' : 'dew_point_min',
                                     'ET0 [result]' : 'eto_result' 
})

weather  = weather.rename(columns = {'Unnamed: 0' :'weather_date'})

print(weather)


# In[6]:


flight_dates = pd.read_excel(df, 'flight dates')

flight_dates  = flight_dates.rename(columns = {'Batch Number' : 'batch_number',
                                               'Flight Date' : 'flight_date'

})

print(flight_dates)


# In[7]:


# check the level of data for flightdates

print(flight_dates.batch_number.nunique())
print(flight_dates.flight_date.nunique())
print(flight_dates.shape[0])

# since both row count and unique values of batch number are same, we can say that the primary key is batch number


# In[8]:


# check the level of data for weather  

print(weather.weather_date.nunique())
print(weather.shape[0])

# we can see that there is some duplication


# In[9]:


type(weather)


# In[10]:


# take only one row of data per weather date 

weather['weather_date_copy'] = weather['weather_date']
weather = weather.groupby('weather_date_copy')
weather = weather.first()
print(weather)


# In[11]:


for col in planting.columns:
    print(col)


# In[12]:


# de duplicating planting dataset

planting['Planting_Batch_copy'] = planting['Planting_Batch']
planting = planting[["Planting_Batch_copy","Planting_Batch","Plant_Date","Region"]]
planting = planting.groupby('Planting_Batch_copy')
planting = planting.first()
print(planting)


# In[13]:


planting = planting[["Planting_Batch","Plant_Date","Region"]]


# In[14]:


print(planting)


# In[15]:


plants = plants[plants["remove"]!='r']
print(plants)


# In[16]:


# Join the plants data and the flights data

plants['batch_number_copy'] = plants['batch_number']
data_1 = plants.set_index('batch_number_copy').join(flight_dates.set_index('batch_number'),lsuffix='', rsuffix='_1')


# In[17]:


data_1.to_csv('data_1.csv')


# In[18]:


for i in plants.columns:
    print(i)


# In[19]:


print(plants.flight_date.unique())


# In[20]:


# combine flight date fields 

data_1['flight_date_final'] = np.where(data_1['flight_date'].isnull(),data_1['flight_date_1'],data_1['flight_date']) 


# In[21]:


# remove nulls

data_2 = data_1[data_1['flight_date_final'].notna()]


# In[22]:


data_2.to_csv('data_2.csv')


# In[23]:


print(data_2)


# In[24]:


# Join the plants data and the flights data

data_2['batch_number_copy2'] = data_2['batch_number']
data_3 = data_2.set_index('batch_number_copy2').join(planting.set_index('Planting_Batch'),lsuffix='', rsuffix='_1')


# In[25]:


print(data_3)


# In[26]:


data_3.to_csv('data_3.csv')


# In[27]:


# remove nulls of plant dates

data_3['plant_date_final'] = np.where(data_3['plant_date'].isnull(),data_3['Plant_Date'],data_3['plant_date']) 


# In[28]:


# remove nulls of head weight

data_4 = data_3[data_3['head_weight'].notna()]


# In[29]:


data_4.isnull().sum()


# In[30]:


# use only required columns in data

data_5 = data_4[['batch_number',
                 'class',
                 'leaves',
                 'leaf_area',
                 'fresh_weight',
                 'radial_diameter',
                 'polar_diameter',
                 'Region',
                 'check_date',
                 'flight_date_final',
                 'plant_date_final']]


# In[31]:


# use date fields to compute days between plant date,check date and flight date

data_5['flight_plant_date_diff'] = (data_5['flight_date_final'] - data_5['plant_date_final'])/np.timedelta64(1,'D')

data_5['check_plant_date_diff'] = (data_5['check_date'] - data_5['plant_date_final'])/np.timedelta64(1,'D')

data_5['check_flight_date_diff'] = (data_5['check_date'] - data_5['flight_date_final'])/np.timedelta64(1,'D')


# In[32]:


data_5.to_csv('data_5.csv')


# In[33]:


print(data_5)


# In[34]:


print(data_5['check_flight_date_diff'].unique())


# In[35]:


# data cleaning to remove records where check date is before flight date

data_6 = data_5[data_5["check_flight_date_diff"] > 0]


# In[36]:


data_6.to_csv('data_6.csv')


# In[37]:


print(data_6)


# In[38]:


print(weather)


# In[39]:


weather['year'] = pd.DatetimeIndex(weather['weather_date']).year
weather['month'] = pd.DatetimeIndex(weather['weather_date']).month
weather['day'] = pd.DatetimeIndex(weather['weather_date']).day
weather['date'] = pd.DatetimeIndex(weather['weather_date']).date


# In[40]:


print(weather)


# In[41]:


# Reduce the weather dataset size. Filter the data from the forst plant date to last check date and keep only 2019 and 2020 years 

plant_date_min = pd.to_datetime(min(data_6['plant_date_final'])).date()
print(plant_date_min)

month_plant_date_min =  pd.to_datetime(plant_date_min).month
print(month_plant_date_min)
day_plant_date_min = pd.to_datetime(plant_date_min).day
print(day_plant_date_min)

check_date_max = pd.to_datetime(max(data_6['check_date'])).date()
print(check_date_max)

month_check_date_max =  pd.to_datetime(check_date_max).month
print(month_check_date_max)
day_check_date_max = pd.to_datetime(check_date_max).day
print(day_check_date_max)

type(check_date_max)


# In[42]:


# Filter the data from the forst plant date to last check date and keep only 2019 and 2020 years 

weather_1 = weather[weather["year"] >= 2019]

weather_1 = weather_1[weather_1["month"] >= month_plant_date_min] 

weather_1 = weather_1[weather_1["month"] <= month_check_date_max] 


# In[43]:


weather_1.to_csv('weather_1.csv')


# In[44]:


print(data_6['check_date'])


# In[45]:


# Adding extra columns for weather related features. These will be updated with actual values in next step

data_7 = data_6

data_7["mean_solar_radiation"] = 0
data_7["mean_precipitation"] = 0
data_7["mean_wind_speed_avg"] = 0
data_7["mean_wind_speed_max"] = 0
data_7["mean_battery_voltage"] = 0
data_7["mean_leaf_wetness"] = 0
data_7["mean_air_temp_avg"] = 0
data_7["mean_air_temp_max"] = 0
data_7["mean_air_temp_min"] = 0
data_7["mean_relative_humidity"] = 0
data_7["mean_dew_point_avg"] = 0
data_7["mean_dew_point_min"] = 0
data_7["mean_eto_result"] = 0

data_7["median_solar_radiation"] = 0
data_7["median_precipitation"] = 0
data_7["median_wind_speed_avg"] = 0
data_7["median_wind_speed_max"] = 0
data_7["median_battery_voltage"] = 0
data_7["median_leaf_wetness"] = 0
data_7["median_air_temp_avg"] = 0
data_7["median_air_temp_max"] = 0
data_7["median_air_temp_min"] = 0
data_7["median_relative_humidity"] = 0
data_7["median_dew_point_avg"] = 0
data_7["median_dew_point_min"] = 0
data_7["median_eto_result"] = 0

data_7["mean_2019_solar_radiation"] = 0
data_7["mean_2019_precipitation"] = 0
data_7["mean_2019_wind_speed_avg"] = 0
data_7["mean_2019_wind_speed_max"] = 0
data_7["mean_2019_battery_voltage"] = 0
data_7["mean_2019_leaf_wetness"] = 0
data_7["mean_2019_air_temp_avg"] = 0
data_7["mean_2019_air_temp_max"] = 0
data_7["mean_2019_air_temp_min"] = 0
data_7["mean_2019_relative_humidity"] = 0
data_7["mean_2019_dew_point_avg"] = 0
data_7["mean_2019_dew_point_min"] = 0
data_7["mean_2019_eto_result"] = 0

data_7["median_2019_solar_radiation"] = 0
data_7["median_2019_precipitation"] = 0
data_7["median_2019_wind_speed_avg"] = 0
data_7["median_2019_wind_speed_max"] = 0
data_7["median_2019_battery_voltage"] = 0
data_7["median_2019_leaf_wetness"] = 0
data_7["median_2019_air_temp_avg"] = 0
data_7["median_2019_air_temp_max"] = 0
data_7["median_2019_air_temp_min"] = 0
data_7["median_2019_relative_humidity"] = 0
data_7["median_2019_dew_point_avg"] = 0
data_7["median_2019_dew_point_min"] = 0
data_7["median_2019_eto_result"] = 0


data_7["mean_2_2019_solar_radiation"] = 0
data_7["mean_2_2019_precipitation"] = 0
data_7["mean_2_2019_wind_speed_avg"] = 0
data_7["mean_2_2019_wind_speed_max"] = 0
data_7["mean_2_2019_battery_voltage"] = 0
data_7["mean_2_2019_leaf_wetness"] = 0
data_7["mean_2_2019_air_temp_avg"] = 0
data_7["mean_2_2019_air_temp_max"] = 0
data_7["mean_2_2019_air_temp_min"] = 0
data_7["mean_2_2019_relative_humidity"] = 0
data_7["mean_2_2019_dew_point_avg"] = 0
data_7["mean_2_2019_dew_point_min"] = 0
data_7["mean_2_2019_eto_result"] = 0

data_7["median_2_2019_solar_radiation"] = 0
data_7["median_2_2019_precipitation"] = 0
data_7["median_2_2019_wind_speed_avg"] = 0
data_7["median_2_2019_wind_speed_max"] = 0
data_7["median_2_2019_battery_voltage"] = 0
data_7["median_2_2019_leaf_wetness"] = 0
data_7["median_2_2019_air_temp_avg"] = 0
data_7["median_2_2019_air_temp_max"] = 0
data_7["median_2_2019_air_temp_min"] = 0
data_7["median_2_2019_relative_humidity"] = 0
data_7["median_2_2019_dew_point_avg"] = 0
data_7["median_2_2019_dew_point_min"] = 0
data_7["median_2_2019_eto_result"] = 0


data_7.to_csv('data_7.csv')


# In[46]:


# For every record of data, compute mean and median values of all weather conditions over the growth cycle of the crop

for i in range(data_7.shape[0]):
    
    # subset the weather data from plant date to flight date and flight date to check date
    
    start_date = data_6['plant_date_final'].iloc[i]
    end_date   = data_6['flight_date_final'].iloc[i]
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    
    s1 = str(start_date)
    s1 = s1[-6:]
    start_date_2019 = ('2019'+s1)
    start_date_2019 = pd.to_datetime(start_date_2019).date()
    
    s2 = str(end_date)
    s2 = s2[-6:]
    end_date_2019 = ('2019'+s2)
    end_date_2019 = pd.to_datetime(end_date_2019).date()
    
    start_date_2 = data_6['flight_date_final'].iloc[i]
    end_date_2   = data_6['check_date'].iloc[i]
    start_date_2 = pd.to_datetime(start_date_2).date()
    end_date_2 = pd.to_datetime(end_date_2).date()
    
    s3 = str(start_date_2)
    s3 = s3[-6:]
    start_date_2019_2 = ('2019'+s3)
    start_date_2019_2 = pd.to_datetime(start_date_2019_2).date()
    
    s4 = str(end_date_2)
    s4 = s4[-6:]
    end_date_2019_2 = ('2019'+s4)
    end_date_2019_2 = pd.to_datetime(end_date_2019_2).date()
    
    
    weather_1["weather_date"] = pd.DatetimeIndex(weather_1["weather_date"]).date
    
    weather_subset = weather_1[ weather_1["weather_date"] >= start_date]
    weather_subset = weather_subset[weather_subset["weather_date"] <= end_date]
    
    weather_subset_2019 = weather_1[weather_1["weather_date"] >= start_date_2019]
    weather_subset_2019 = weather_subset_2019[weather_subset_2019["weather_date"] <= end_date_2019]
    
    weather_subset_2019_2 = weather_1[weather_1["weather_date"] >= start_date_2019_2]
    weather_subset_2019_2 = weather_subset_2019_2[weather_subset_2019_2["weather_date"] <= end_date_2019_2]
    
    # compute the mean and median values of all weather related information : 
    
    # for plant date to flight date use both 2019 and 2020
    
    # for flight date to check date use only 2019 (no data leakage)
    
    temp_mean = weather_subset[['solar_radiation',
                          'precipitation',
                          'wind_speed_avg',
                          'wind_speed_max',
                          'battery_voltage',
                          'leaf_wetness',
                          'air_temp_avg',
                          'air_temp_max',
                          'air_temp_min',
                          'relative_humidity',
                          'dew_point_avg',
                          'dew_point_min',
                          'eto_result']].mean()

    temp_median = weather_subset[['solar_radiation',
                          'precipitation',
                          'wind_speed_avg',
                          'wind_speed_max',
                          'battery_voltage',
                          'leaf_wetness',
                          'air_temp_avg',
                          'air_temp_max',
                          'air_temp_min',
                          'relative_humidity',
                          'dew_point_avg',
                          'dew_point_min',
                          'eto_result']].median()
    
    temp_mean_2019 = weather_subset_2019[['solar_radiation',
                          'precipitation',
                          'wind_speed_avg',
                          'wind_speed_max',
                          'battery_voltage',
                          'leaf_wetness',
                          'air_temp_avg',
                          'air_temp_max',
                          'air_temp_min',
                          'relative_humidity',
                          'dew_point_avg',
                          'dew_point_min',
                          'eto_result']].mean()

    temp_median_2019 = weather_subset_2019[['solar_radiation',
                          'precipitation',
                          'wind_speed_avg',
                          'wind_speed_max',
                          'battery_voltage',
                          'leaf_wetness',
                          'air_temp_avg',
                          'air_temp_max',
                          'air_temp_min',
                          'relative_humidity',
                          'dew_point_avg',
                          'dew_point_min',
                          'eto_result']].median()

    temp_mean_2019_2 = weather_subset_2019_2[['solar_radiation',
                          'precipitation',
                          'wind_speed_avg',
                          'wind_speed_max',
                          'battery_voltage',
                          'leaf_wetness',
                          'air_temp_avg',
                          'air_temp_max',
                          'air_temp_min',
                          'relative_humidity',
                          'dew_point_avg',
                          'dew_point_min',
                          'eto_result']].mean()

    temp_median_2019_2 = weather_subset_2019_2[['solar_radiation',
                          'precipitation',
                          'wind_speed_avg',
                          'wind_speed_max',
                          'battery_voltage',
                          'leaf_wetness',
                          'air_temp_avg',
                          'air_temp_max',
                          'air_temp_min',
                          'relative_humidity',
                          'dew_point_avg',
                          'dew_point_min',
                          'eto_result']].median()

    # update the rows with computed values
    
    data_7['mean_solar_radiation'  ].iloc[i] = temp_mean[0]
    data_7['mean_precipitation'    ].iloc[i] = temp_mean[1]
    data_7['mean_wind_speed_avg'   ].iloc[i] = temp_mean[2]
    data_7['mean_wind_speed_max'   ].iloc[i] = temp_mean[3]
    data_7['mean_battery_voltage'  ].iloc[i] = temp_mean[4]
    data_7['mean_leaf_wetness'     ].iloc[i] = temp_mean[5]
    data_7['mean_air_temp_avg'     ].iloc[i] = temp_mean[6]
    data_7['mean_air_temp_max'     ].iloc[i] = temp_mean[7]
    data_7['mean_air_temp_min'     ].iloc[i] = temp_mean[8]
    data_7['mean_relative_humidity'].iloc[i] = temp_mean[9]
    data_7['mean_dew_point_avg'    ].iloc[i] = temp_mean[10]
    data_7['mean_dew_point_min'    ].iloc[i] = temp_mean[11]
    data_7['mean_eto_result'       ].iloc[i] = temp_mean[12]
    
    data_7['median_solar_radiation'  ].iloc[i] = temp_median[0]
    data_7['median_precipitation'    ].iloc[i] = temp_median[1]
    data_7['median_wind_speed_avg'   ].iloc[i] = temp_median[2]
    data_7['median_wind_speed_max'   ].iloc[i] = temp_median[3]
    data_7['median_battery_voltage'  ].iloc[i] = temp_median[4]
    data_7['median_leaf_wetness'     ].iloc[i] = temp_median[5]
    data_7['median_air_temp_avg'     ].iloc[i] = temp_median[6]
    data_7['median_air_temp_max'     ].iloc[i] = temp_median[7]
    data_7['median_air_temp_min'     ].iloc[i] = temp_median[8]
    data_7['median_relative_humidity'].iloc[i] = temp_median[9]
    data_7['median_dew_point_avg'    ].iloc[i] = temp_median[10]
    data_7['median_dew_point_min'    ].iloc[i] = temp_median[11]
    data_7['median_eto_result'       ].iloc[i] = temp_median[12]

    data_7['mean_2019_solar_radiation'  ].iloc[i] = temp_mean_2019[0]
    data_7['mean_2019_precipitation'    ].iloc[i] = temp_mean_2019[1]
    data_7['mean_2019_wind_speed_avg'   ].iloc[i] = temp_mean_2019[2]
    data_7['mean_2019_wind_speed_max'   ].iloc[i] = temp_mean_2019[3]
    data_7['mean_2019_battery_voltage'  ].iloc[i] = temp_mean_2019[4]
    data_7['mean_2019_leaf_wetness'     ].iloc[i] = temp_mean_2019[5]
    data_7['mean_2019_air_temp_avg'     ].iloc[i] = temp_mean_2019[6]
    data_7['mean_2019_air_temp_max'     ].iloc[i] = temp_mean_2019[7]
    data_7['mean_2019_air_temp_min'     ].iloc[i] = temp_mean_2019[8]
    data_7['mean_2019_relative_humidity'].iloc[i] = temp_mean_2019[9]
    data_7['mean_2019_dew_point_avg'    ].iloc[i] = temp_mean_2019[10]
    data_7['mean_2019_dew_point_min'    ].iloc[i] = temp_mean_2019[11]
    data_7['mean_2019_eto_result'       ].iloc[i] = temp_mean_2019[12]
    
    data_7['median_2019_solar_radiation'  ].iloc[i] = temp_median_2019[0]
    data_7['median_2019_precipitation'    ].iloc[i] = temp_median_2019[1]
    data_7['median_2019_wind_speed_avg'   ].iloc[i] = temp_median_2019[2]
    data_7['median_2019_wind_speed_max'   ].iloc[i] = temp_median_2019[3]
    data_7['median_2019_battery_voltage'  ].iloc[i] = temp_median_2019[4]
    data_7['median_2019_leaf_wetness'     ].iloc[i] = temp_median_2019[5]
    data_7['median_2019_air_temp_avg'     ].iloc[i] = temp_median_2019[6]
    data_7['median_2019_air_temp_max'     ].iloc[i] = temp_median_2019[7]
    data_7['median_2019_air_temp_min'     ].iloc[i] = temp_median_2019[8]
    data_7['median_2019_relative_humidity'].iloc[i] = temp_median_2019[9]
    data_7['median_2019_dew_point_avg'    ].iloc[i] = temp_median_2019[10]
    data_7['median_2019_dew_point_min'    ].iloc[i] = temp_median_2019[11]
    data_7['median_2019_eto_result'       ].iloc[i] = temp_median_2019[12]
    
    data_7['mean_2_2019_solar_radiation'  ].iloc[i] = temp_mean_2019_2[0]
    data_7['mean_2_2019_precipitation'    ].iloc[i] = temp_mean_2019_2[1]
    data_7['mean_2_2019_wind_speed_avg'   ].iloc[i] = temp_mean_2019_2[2]
    data_7['mean_2_2019_wind_speed_max'   ].iloc[i] = temp_mean_2019_2[3]
    data_7['mean_2_2019_battery_voltage'  ].iloc[i] = temp_mean_2019_2[4]
    data_7['mean_2_2019_leaf_wetness'     ].iloc[i] = temp_mean_2019_2[5]
    data_7['mean_2_2019_air_temp_avg'     ].iloc[i] = temp_mean_2019_2[6]
    data_7['mean_2_2019_air_temp_max'     ].iloc[i] = temp_mean_2019_2[7]
    data_7['mean_2_2019_air_temp_min'     ].iloc[i] = temp_mean_2019_2[8]
    data_7['mean_2_2019_relative_humidity'].iloc[i] = temp_mean_2019_2[9]
    data_7['mean_2_2019_dew_point_avg'    ].iloc[i] = temp_mean_2019_2[10]
    data_7['mean_2_2019_dew_point_min'    ].iloc[i] = temp_mean_2019_2[11]
    data_7['mean_2_2019_eto_result'       ].iloc[i] = temp_mean_2019_2[12]
    
    data_7['median_2_2019_solar_radiation'  ].iloc[i] = temp_median_2019_2[0]
    data_7['median_2_2019_precipitation'    ].iloc[i] = temp_median_2019_2[1]
    data_7['median_2_2019_wind_speed_avg'   ].iloc[i] = temp_median_2019_2[2]
    data_7['median_2_2019_wind_speed_max'   ].iloc[i] = temp_median_2019_2[3]
    data_7['median_2_2019_battery_voltage'  ].iloc[i] = temp_median_2019_2[4]
    data_7['median_2_2019_leaf_wetness'     ].iloc[i] = temp_median_2019_2[5]
    data_7['median_2_2019_air_temp_avg'     ].iloc[i] = temp_median_2019_2[6]
    data_7['median_2_2019_air_temp_max'     ].iloc[i] = temp_median_2019_2[7]
    data_7['median_2_2019_air_temp_min'     ].iloc[i] = temp_median_2019_2[8]
    data_7['median_2_2019_relative_humidity'].iloc[i] = temp_median_2019_2[9]
    data_7['median_2_2019_dew_point_avg'    ].iloc[i] = temp_median_2019_2[10]
    data_7['median_2_2019_dew_point_min'    ].iloc[i] = temp_median_2019_2[11]
    data_7['median_2_2019_eto_result'       ].iloc[i] = temp_median_2019_2[12]


# In[47]:


data_7.to_csv('data_7.csv')


# # Plots and Tables to undertand the final dataset

# In[52]:


# Distribution of weight

df = data_7['fresh_weight']
df.quantile([0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1])


# In[53]:


df.mean()


# In[48]:


# Distribution of fresh weight of plants. 

plt.rcParams["figure.figsize"] = [7.50, 5]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots()

data_7['fresh_weight_buckets'] = np.where(data_7['fresh_weight']<200, '0 to 200', 
                                 np.where(data_7['fresh_weight']<400, '200 to 400',
                                 np.where(data_7['fresh_weight']<600, '400 to 600',
                                 np.where(data_7['fresh_weight']<800, '600 to 800',
                                 np.where(data_7['fresh_weight']<1000, '800 to 1000',
                                 np.where(data_7['fresh_weight']<1200, '1000 to 1200',
                                 np.where(data_7['fresh_weight']<1400, '1200 to 1400',
                                 np.where(data_7['fresh_weight']<1600, '1400 to 1600',
                                 np.where(data_7['fresh_weight']<1800, '1600 to 1800',
                                 np.where(data_7['fresh_weight']<2000, '1800 to 2000','0'
                                         ))))))))))



data_7['fresh_weight_buckets'].value_counts(normalize=True).plot(ax=ax, kind='bar', xlabel='weight bucket', ylabel='frequency')

plt.show()

# Nearly 84% of the plants have a weight of between 200 grams to 1000 grams, with nearly 21% of plants
# having weights in each of these buckets :'400-600gm','800-1000 gm'
# 25% of the plants weigh between 600 to 800gm. 15% of plants weigh between 200-400 gm.


# In[56]:


# Distribution of polar diameter

df = data_7['polar_diameter']
print(df.quantile([0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1]))
print(df.mean())


# In[49]:


# Distribution of polar diameter of plants. 

plt.rcParams["figure.figsize"] = [7.50,5]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots()

data_7['polar_buckets'] = np.where(data_7['polar_diameter']<20, '0 to 20', 
                          np.where(data_7['polar_diameter']<40, '20 to 40',
                          np.where(data_7['polar_diameter']<60, '40 to 60',
                          np.where(data_7['polar_diameter']<80, '60 to 80',
                          np.where(data_7['polar_diameter']<100, '80 to 100',
                          np.where(data_7['polar_diameter']<120, '100 to 120',
                          np.where(data_7['polar_diameter']<140, '120 to 140',
                          np.where(data_7['polar_diameter']<160, '140 to 160',
                          np.where(data_7['polar_diameter']<180, '160 to 180',
                          np.where(data_7['polar_diameter']<200, '180 to 200','0'
                                         ))))))))))



data_7['polar_buckets'].value_counts(normalize=True).plot(ax=ax, kind='bar', xlabel='polar diameter bucket', ylabel='frequency')

plt.show()

# Nearly one-third of the plants have a polar diameter of between 120 to 140 mm, with nearly 85% of plants
# have a polar diameter of 80-160 mm


# In[54]:


# Distribution of radial diameter

df = data_7['radial_diameter']
print(df.quantile([0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1]))
print(df.mean())


# In[50]:


# Distribution of radial diameter of plants. 

plt.rcParams["figure.figsize"] = [7.50,5]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots()

data_7['radial_buckets'] = np.where(data_7['radial_diameter']<20, '0 to 20', 
                           np.where(data_7['radial_diameter']<40, '20 to 40',
                           np.where(data_7['radial_diameter']<60, '40 to 60',
                           np.where(data_7['radial_diameter']<80, '60 to 80',
                           np.where(data_7['radial_diameter']<100, '80 to 100',
                           np.where(data_7['radial_diameter']<120, '100 to 120',
                           np.where(data_7['radial_diameter']<140, '120 to 140',
                           np.where(data_7['radial_diameter']<160, '140 to 160',
                           np.where(data_7['radial_diameter']<180, '160 to 180',
                           np.where(data_7['radial_diameter']<200, '180 to 200','0'
                                         ))))))))))



data_7['radial_buckets'].value_counts(normalize=True).plot(ax=ax, kind='bar', xlabel='radial diameter bucket', ylabel='frequency')

plt.show()

# Nearly half of the plants have a radial diameter of between 120 to 160 mm


# In[51]:


# Distribution of weight split by class
# Does weight increase with increasing class ?

data_plot = data_7[['class','fresh_weight_buckets','check_date']]

table = pd.pivot_table(data_plot, values=['check_date'], index=['fresh_weight_buckets'],
                    columns=['class'], aggfunc='count')

table


# In[52]:


# correlation between class and weight

data_7['class'].corr(data_7['fresh_weight'])


# In[53]:


# Distribution of weight split by leaf area
# Does weight increase with increasing leaf area ?

data_plot = data_7[['leaf_area','fresh_weight_buckets','check_date']]

data_plot['leafarea_buckets'] = np.where(data_7['leaf_area']<200, '00 to 200', 
                                np.where(data_7['leaf_area']<400, '200 to 400',
                                np.where(data_7['leaf_area']<600, '400 to 600',
                                np.where(data_7['leaf_area']<800, '600 to 800',
                                np.where(data_7['leaf_area']<1000, '800 to 1000',
                                np.where(data_7['leaf_area']<1200, '1000 to 1200',
                                np.where(data_7['leaf_area']<1400, '1200 to 1400','0'
                                         )))))))

table = pd.pivot_table(data_plot, values=['check_date'], index=['fresh_weight_buckets'],
                    columns=['leafarea_buckets'], aggfunc='count')

table


# In[54]:


# correlation between leaf area and weight

data_7['leaf_area'].corr(data_7['fresh_weight'])


# In[55]:


# Distribution of weight split by growth cycle (difference between check date and plant date)
# Does weight increase with increasing growth cycle ?

data_plot = data_7[['check_plant_date_diff','fresh_weight_buckets','check_date']]

data_plot['check_plant_date_diff_bucket'] = np.where(data_7['check_plant_date_diff']<35, '0 to 35', 
                                            np.where(data_7['check_plant_date_diff']<40, '35 to 40',
                                            np.where(data_7['check_plant_date_diff']<45, '40 to 45',
                                            np.where(data_7['check_plant_date_diff']<50, '45 to 50',
                                            np.where(data_7['check_plant_date_diff']<55, '50 to 55','0'
                                         )))))

table = pd.pivot_table(data_plot, values=['check_date'], index=['fresh_weight_buckets'],
                    columns=['check_plant_date_diff_bucket'], aggfunc='count')

table


# In[ ]:





# # Creating train and test datasets

# In[59]:


# features dataset 

X=data_7[['batch_number','class','leaf_area','Region','flight_plant_date_diff','check_plant_date_diff','check_flight_date_diff','check_date','plant_date_final','flight_date_final','mean_solar_radiation','mean_precipitation','mean_wind_speed_avg','mean_wind_speed_max','mean_battery_voltage','mean_leaf_wetness','mean_air_temp_avg','mean_air_temp_max','mean_air_temp_min','mean_relative_humidity','mean_dew_point_avg','mean_dew_point_min','mean_eto_result','median_solar_radiation','median_precipitation','median_wind_speed_avg','median_wind_speed_max','median_battery_voltage','median_leaf_wetness','median_air_temp_avg','median_air_temp_max','median_air_temp_min','median_relative_humidity','median_dew_point_avg','median_dew_point_min','median_eto_result','mean_2019_solar_radiation','mean_2019_precipitation','mean_2019_wind_speed_avg','mean_2019_wind_speed_max','mean_2019_battery_voltage','mean_2019_leaf_wetness','mean_2019_air_temp_avg','mean_2019_air_temp_max','mean_2019_air_temp_min','mean_2019_relative_humidity','mean_2019_dew_point_avg','mean_2019_dew_point_min','mean_2019_eto_result','median_2019_solar_radiation','median_2019_precipitation','median_2019_wind_speed_avg','median_2019_wind_speed_max','median_2019_battery_voltage','median_2019_leaf_wetness','median_2019_air_temp_avg','median_2019_air_temp_max','median_2019_air_temp_min','median_2019_relative_humidity','median_2019_dew_point_avg','median_2019_dew_point_min','median_2019_eto_result','mean_2_2019_solar_radiation','mean_2_2019_precipitation','mean_2_2019_wind_speed_avg','mean_2_2019_wind_speed_max','mean_2_2019_battery_voltage','mean_2_2019_leaf_wetness','mean_2_2019_air_temp_avg','mean_2_2019_air_temp_max','mean_2_2019_air_temp_min','mean_2_2019_relative_humidity','mean_2_2019_dew_point_avg','mean_2_2019_dew_point_min','mean_2_2019_eto_result','median_2_2019_solar_radiation','median_2_2019_precipitation','median_2_2019_wind_speed_avg','median_2_2019_wind_speed_max','median_2_2019_battery_voltage','median_2_2019_leaf_wetness','median_2_2019_air_temp_avg','median_2_2019_air_temp_max','median_2_2019_air_temp_min','median_2_2019_relative_humidity','median_2_2019_dew_point_avg','median_2_2019_dew_point_min','median_2_2019_eto_result'
         ]]

# target dataset

y = data_7[['fresh_weight','radial_diameter','polar_diameter']]


# In[60]:


print(X)


# In[61]:


print(y)


# In[62]:


# create train and test datasets

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,random_state=12,stratify = data_7["class"])

# keep a copy as we need check dates to see how much weight to expect each day (graph to plot sum of lettuce weight vs check date for X_test dataset)

X_test_withdate = X_test 
X_test_withdate.to_csv('X_test_withdate.csv')

# drop date variables as regressor cant handle date columns

X_train =  X_train.drop(['check_date' , 'plant_date_final','flight_date_final'],axis=1)
X_test = X_test.drop(['check_date' , 'plant_date_final','flight_date_final'],axis=1)


# In[63]:


print(X_train.shape[0])
print(X_train.shape[1])
print(X_test.shape[0])
print(X_test.shape[1])

print(y_train.shape[0])
print(y_train.shape[1])
print(y_test.shape[0])
print(y_test.shape[1])

X_train.to_csv('X_train.csv')
y_train.to_csv('y_train.csv')


# # Random Forest Regressor

# In[64]:


# random forest model

clf = MultiOutputRegressor(RandomForestRegressor(max_depth=5, bootstrap = True, max_features= 'auto',n_estimators = 100,min_samples_leaf = 3,min_samples_split = 10,random_state=0))
clf.fit(X_train, y_train)


# In[65]:


clf.score(X_test, y_test, sample_weight=None)


# In[66]:


clf_grid = MultiOutputRegressor(RandomForestRegressor())


# In[67]:


# cross validation

cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=5)
n_scores = cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
n_scores = absolute(n_scores)

print('MAE: %.3f' % (mean(n_scores)))


# In[65]:


# create a grid of parameters 

random_grid = {'estimator__n_estimators':[100,150,200],
               'estimator__max_features': ['sqrt','auto'],
               'estimator__max_depth': [3,4,5]}


# In[66]:


# search for the best parameters

rf_grid = GridSearchCV(clf_grid,  random_grid, cv = 3)

rf_grid = rf_grid.fit(X_train, y_train)


# In[67]:


rf_grid.best_params_


# In[68]:


rf_grid.score(X_test, y_test)


# In[78]:


# Use dummy regressor to compare

dummy_regressor = DummyRegressor(strategy="mean")
dummy_regressor.fit(X_train, y_train)


# In[79]:


# check performance of dummy regressor 

dummy_regressor.score(X_train, y_train)


# In[69]:


cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=5)
n_scores = cross_val_score(rf_grid, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
n_scores = absolute(n_scores)

print('MAE: %.3f' % (mean(n_scores)))


# In[73]:


# predict on test data

ypred_rf = clf.predict(X_test)
ypred_rf = pd.DataFrame(ypred_rf)
print(ypred_rf)


ypred_rf = ypred_rf.reset_index(drop=True)
ypred_rf.to_csv('ypred_rf.csv')
X_test_withdate = X_test_withdate.reset_index(drop=True)
X_test_withdate.to_csv('X_test_withdate.csv')

# merge the predictions with input variables

rf_merged = pd.concat([X_test_withdate,ypred_rf],axis=1,ignore_index=True)


rf_merged = rf_merged.rename(columns={0:'batch_number',	1:'class',	2:'leaf_area',	3:'Region',	4:'flight_plant_date_diff',	5:'check_plant_date_diff',	6:'check_flight_date_diff',	7:'check_date',	8:'plant_date_final',	9:'flight_date_final',	10:'mean_solar_radiation',	11:'mean_precipitation',	12:'mean_wind_speed_avg',	13:'mean_wind_speed_max',	14:'mean_battery_voltage',	15:'mean_leaf_wetness',	16:'mean_air_temp_avg',	17:'mean_air_temp_max',	18:'mean_air_temp_min',	19:'mean_relative_humidity',	20:'mean_dew_point_avg',	21:'mean_dew_point_min',	22:'mean_eto_result',	23:'median_solar_radiation',	24:'median_precipitation',	25:'median_wind_speed_avg',	26:'median_wind_speed_max',	27:'median_battery_voltage',	28:'median_leaf_wetness',	29:'median_air_temp_avg',	30:'median_air_temp_max',	31:'median_air_temp_min',	32:'median_relative_humidity',	33:'median_dew_point_avg',	34:'median_dew_point_min',	35:'median_eto_result',	36:'mean_2019_solar_radiation',	37:'mean_2019_precipitation',	38:'mean_2019_wind_speed_avg',	39:'mean_2019_wind_speed_max',	40:'mean_2019_battery_voltage',	41:'mean_2019_leaf_wetness',	42:'mean_2019_air_temp_avg',	43:'mean_2019_air_temp_max',	44:'mean_2019_air_temp_min',	45:'mean_2019_relative_humidity',	46:'mean_2019_dew_point_avg',	47:'mean_2019_dew_point_min',	48:'mean_2019_eto_result',	49:'median_2019_solar_radiation',	50:'median_2019_precipitation',	51:'median_2019_wind_speed_avg',	52:'median_2019_wind_speed_max',	53:'median_2019_battery_voltage',	54:'median_2019_leaf_wetness',	55:'median_2019_air_temp_avg',	56:'median_2019_air_temp_max',	57:'median_2019_air_temp_min',	58:'median_2019_relative_humidity',	59:'median_2019_dew_point_avg',	60:'median_2019_dew_point_min',	61:'median_2019_eto_result',	62:'mean_2_2019_solar_radiation',	63:'mean_2_2019_precipitation',	64:'mean_2_2019_wind_speed_avg',	65:'mean_2_2019_wind_speed_max',	66:'mean_2_2019_battery_voltage',	67:'mean_2_2019_leaf_wetness',	68:'mean_2_2019_air_temp_avg',	69:'mean_2_2019_air_temp_max',	70:'mean_2_2019_air_temp_min',	71:'mean_2_2019_relative_humidity',	72:'mean_2_2019_dew_point_avg',	73:'mean_2_2019_dew_point_min',	74:'mean_2_2019_eto_result',	75:'median_2_2019_solar_radiation',	76:'median_2_2019_precipitation',	77:'median_2_2019_wind_speed_avg',	78:'median_2_2019_wind_speed_max',	79:'median_2_2019_battery_voltage',	80:'median_2_2019_leaf_wetness',	81:'median_2_2019_air_temp_avg',	82:'median_2_2019_air_temp_max',	83:'median_2_2019_air_temp_min',	84:'median_2_2019_relative_humidity',	85:'median_2_2019_dew_point_avg',	86:'median_2_2019_dew_point_min',	87:'median_2_2019_eto_result',
                                      88:'fresh_weight',	89:'radial_diameter',	90:'polar_diameter'
})
print(rf_merged)
rf_merged.to_csv('rf_merged.csv')


# In[75]:


# Analyse the predictions : How much lettuce is expected each day ??


rf_merged_1 = rf_merged[['check_date','fresh_weight']]
rf_merged_2 = rf_merged_1.groupby(['check_date'],sort=True).sum()
rf_merged_2 = rf_merged_2.sort_values(by=['check_date'])


rf_merged_1 = rf_merged[['check_date']]
rf_merged_1['check_date_copy'] = rf_merged_1['check_date']
rf_merged_3 = rf_merged_1.groupby(['check_date_copy'],sort=True).max()
rf_merged_3 = rf_merged_3.sort_values(by=['check_date'])


rf_merged_2 = rf_merged_2.reset_index(drop=True)
rf_merged_3 = rf_merged_3.reset_index(drop=True)

rf_merged_4 = pd.concat([rf_merged_2,rf_merged_3],axis=1,ignore_index=True)
rf_merged_4 = rf_merged_4.rename(columns={0:'sum_fresh_weight',	1:'check_date'})
rf_merged_4 = rf_merged_4[['check_date','sum_fresh_weight']]
rf_merged_4


# In[89]:


# Analyse the predictions : How much lettuce is expected each month ??


rf_merged_1 = rf_merged[['check_date','fresh_weight']]
rf_merged_1['month'] = pd.DatetimeIndex(rf_merged_1['check_date']).month
rf_merged_1.drop(columns = ['check_date'])
rf_merged_2 = rf_merged_1.groupby(['month'],sort=True).sum()
rf_merged_2 = rf_merged_2.sort_values(by=['month'])


rf_merged_1 = rf_merged[['check_date']]
rf_merged_1['month'] = pd.DatetimeIndex(rf_merged_1['check_date']).month
rf_merged_1['month_copy'] = rf_merged_1['month']
rf_merged_1.drop(columns = ['check_date'])
rf_merged_0 = rf_merged_1[['month','month_copy']]
rf_merged_3 = rf_merged_0.groupby(['month_copy'],sort=True).max()
rf_merged_3 = rf_merged_3.sort_values(by=['month'])


rf_merged_2 = rf_merged_2.reset_index(drop=True)
rf_merged_3 = rf_merged_3.reset_index(drop=True)
rf_merged_4 = pd.concat([rf_merged_2,rf_merged_3],axis=1,ignore_index=True)
rf_merged_4 = rf_merged_4.rename(columns={0:'sum_fresh_weight',	1:'month'})
rf_merged_4 = rf_merged_4[['month','sum_fresh_weight']]
rf_merged_4


# In[72]:


# plot of expected lettuce produce over coming days

plt.plot(rf_merged_4['check_date'],rf_merged_4['sum_fresh_weight'])
plt.rcParams["figure.figsize"] = [100,5]
plt.title('Lettuce Produce over Days')
plt.xlabel('Date')
plt.ylabel('Weight of Produce')
plt.show()


# In[73]:


# Which region has highest weight of lettuce ?

rf_merged_1 = rf_merged[['Region','fresh_weight']]
rf_merged_2 = rf_merged_1.groupby(['Region'],sort=True).sum()
rf_merged_2

# Since the test data has only one single region, we cant compare weight of produce across different regions.
# With real data we can check for average weight per plant to see which region has highest weight per plant


# In[74]:


# how many plants were planted in each region ?

rf_merged_1 = rf_merged[['Region','fresh_weight']]
rf_merged_2 = rf_merged_1.groupby(['Region'],sort=True).count()
rf_merged_2

# we can take average produce for each region (which is total produce divided by number of plants) and see which region has 
# highest weight per plant. More plants can be planted here as the produce is better than other regions.Weather conditions of 
# this region can be suitable for plants which is resulting in better produce


# In[75]:


# Check feature importances for fresh weight

features_fresh_weight = clf.estimators_[0].feature_importances_
column_names =['batch_number','class','leaf_area','Region','flight_plant_date_diff','check_plant_date_diff','check_flight_date_diff','mean_solar_radiation','mean_precipitation','mean_wind_speed_avg','mean_wind_speed_max','mean_battery_voltage','mean_leaf_wetness','mean_air_temp_avg','mean_air_temp_max','mean_air_temp_min','mean_relative_humidity','mean_dew_point_avg','mean_dew_point_min','mean_eto_result','median_solar_radiation','median_precipitation','median_wind_speed_avg','median_wind_speed_max','median_battery_voltage','median_leaf_wetness','median_air_temp_avg','median_air_temp_max','median_air_temp_min','median_relative_humidity','median_dew_point_avg','median_dew_point_min','median_eto_result','mean_2019_solar_radiation','mean_2019_precipitation','mean_2019_wind_speed_avg','mean_2019_wind_speed_max','mean_2019_battery_voltage','mean_2019_leaf_wetness','mean_2019_air_temp_avg','mean_2019_air_temp_max','mean_2019_air_temp_min','mean_2019_relative_humidity','mean_2019_dew_point_avg','mean_2019_dew_point_min','mean_2019_eto_result','median_2019_solar_radiation','median_2019_precipitation','median_2019_wind_speed_avg','median_2019_wind_speed_max','median_2019_battery_voltage','median_2019_leaf_wetness','median_2019_air_temp_avg','median_2019_air_temp_max','median_2019_air_temp_min','median_2019_relative_humidity','median_2019_dew_point_avg','median_2019_dew_point_min','median_2019_eto_result','mean_2_2019_solar_radiation','mean_2_2019_precipitation','mean_2_2019_wind_speed_avg','mean_2_2019_wind_speed_max','mean_2_2019_battery_voltage','mean_2_2019_leaf_wetness','mean_2_2019_air_temp_avg','mean_2_2019_air_temp_max','mean_2_2019_air_temp_min','mean_2_2019_relative_humidity','mean_2_2019_dew_point_avg','mean_2_2019_dew_point_min','mean_2_2019_eto_result','median_2_2019_solar_radiation','median_2_2019_precipitation','median_2_2019_wind_speed_avg','median_2_2019_wind_speed_max','median_2_2019_battery_voltage','median_2_2019_leaf_wetness','median_2_2019_air_temp_avg','median_2_2019_air_temp_max','median_2_2019_air_temp_min','median_2_2019_relative_humidity','median_2_2019_dew_point_avg','median_2_2019_dew_point_min','median_2_2019_eto_result']
fresh_weight_col = pd.DataFrame(columns=['feat', 'colname'])

fresh_weight_col['feat'] = features_fresh_weight
fresh_weight_col['colname']='0'
for i in range(len(column_names)):
    fresh_weight_col['colname'].iloc[i] = column_names[i]



fresh_weight_col = fresh_weight_col[['colname','feat']]

pd.set_option('display.max_rows', 100)
print(fresh_weight_col.sort_values(by=['feat'], ascending=False))


# In[76]:


# Check feature importances for radial diameter

features_fresh_weight =clf.estimators_[1].feature_importances_
column_names =['batch_number','class','leaf_area','Region','flight_plant_date_diff','check_plant_date_diff','check_flight_date_diff','mean_solar_radiation','mean_precipitation','mean_wind_speed_avg','mean_wind_speed_max','mean_battery_voltage','mean_leaf_wetness','mean_air_temp_avg','mean_air_temp_max','mean_air_temp_min','mean_relative_humidity','mean_dew_point_avg','mean_dew_point_min','mean_eto_result','median_solar_radiation','median_precipitation','median_wind_speed_avg','median_wind_speed_max','median_battery_voltage','median_leaf_wetness','median_air_temp_avg','median_air_temp_max','median_air_temp_min','median_relative_humidity','median_dew_point_avg','median_dew_point_min','median_eto_result','mean_2019_solar_radiation','mean_2019_precipitation','mean_2019_wind_speed_avg','mean_2019_wind_speed_max','mean_2019_battery_voltage','mean_2019_leaf_wetness','mean_2019_air_temp_avg','mean_2019_air_temp_max','mean_2019_air_temp_min','mean_2019_relative_humidity','mean_2019_dew_point_avg','mean_2019_dew_point_min','mean_2019_eto_result','median_2019_solar_radiation','median_2019_precipitation','median_2019_wind_speed_avg','median_2019_wind_speed_max','median_2019_battery_voltage','median_2019_leaf_wetness','median_2019_air_temp_avg','median_2019_air_temp_max','median_2019_air_temp_min','median_2019_relative_humidity','median_2019_dew_point_avg','median_2019_dew_point_min','median_2019_eto_result','mean_2_2019_solar_radiation','mean_2_2019_precipitation','mean_2_2019_wind_speed_avg','mean_2_2019_wind_speed_max','mean_2_2019_battery_voltage','mean_2_2019_leaf_wetness','mean_2_2019_air_temp_avg','mean_2_2019_air_temp_max','mean_2_2019_air_temp_min','mean_2_2019_relative_humidity','mean_2_2019_dew_point_avg','mean_2_2019_dew_point_min','mean_2_2019_eto_result','median_2_2019_solar_radiation','median_2_2019_precipitation','median_2_2019_wind_speed_avg','median_2_2019_wind_speed_max','median_2_2019_battery_voltage','median_2_2019_leaf_wetness','median_2_2019_air_temp_avg','median_2_2019_air_temp_max','median_2_2019_air_temp_min','median_2_2019_relative_humidity','median_2_2019_dew_point_avg','median_2_2019_dew_point_min','median_2_2019_eto_result']
fresh_weight_col = pd.DataFrame(columns=['feat', 'colname'])

fresh_weight_col['feat'] = features_fresh_weight
fresh_weight_col['colname']='0'
for i in range(len(column_names)):
    fresh_weight_col['colname'].iloc[i] = column_names[i]



fresh_weight_col = fresh_weight_col[['colname','feat']]

pd.set_option('display.max_rows', 100)
print(fresh_weight_col.sort_values(by=['feat'], ascending=False))


# In[77]:


# Check feature importances for polar diameter

features_fresh_weight = clf.estimators_[2].feature_importances_
column_names =['batch_number','class','leaf_area','Region','flight_plant_date_diff','check_plant_date_diff','check_flight_date_diff','mean_solar_radiation','mean_precipitation','mean_wind_speed_avg','mean_wind_speed_max','mean_battery_voltage','mean_leaf_wetness','mean_air_temp_avg','mean_air_temp_max','mean_air_temp_min','mean_relative_humidity','mean_dew_point_avg','mean_dew_point_min','mean_eto_result','median_solar_radiation','median_precipitation','median_wind_speed_avg','median_wind_speed_max','median_battery_voltage','median_leaf_wetness','median_air_temp_avg','median_air_temp_max','median_air_temp_min','median_relative_humidity','median_dew_point_avg','median_dew_point_min','median_eto_result','mean_2019_solar_radiation','mean_2019_precipitation','mean_2019_wind_speed_avg','mean_2019_wind_speed_max','mean_2019_battery_voltage','mean_2019_leaf_wetness','mean_2019_air_temp_avg','mean_2019_air_temp_max','mean_2019_air_temp_min','mean_2019_relative_humidity','mean_2019_dew_point_avg','mean_2019_dew_point_min','mean_2019_eto_result','median_2019_solar_radiation','median_2019_precipitation','median_2019_wind_speed_avg','median_2019_wind_speed_max','median_2019_battery_voltage','median_2019_leaf_wetness','median_2019_air_temp_avg','median_2019_air_temp_max','median_2019_air_temp_min','median_2019_relative_humidity','median_2019_dew_point_avg','median_2019_dew_point_min','median_2019_eto_result','mean_2_2019_solar_radiation','mean_2_2019_precipitation','mean_2_2019_wind_speed_avg','mean_2_2019_wind_speed_max','mean_2_2019_battery_voltage','mean_2_2019_leaf_wetness','mean_2_2019_air_temp_avg','mean_2_2019_air_temp_max','mean_2_2019_air_temp_min','mean_2_2019_relative_humidity','mean_2_2019_dew_point_avg','mean_2_2019_dew_point_min','mean_2_2019_eto_result','median_2_2019_solar_radiation','median_2_2019_precipitation','median_2_2019_wind_speed_avg','median_2_2019_wind_speed_max','median_2_2019_battery_voltage','median_2_2019_leaf_wetness','median_2_2019_air_temp_avg','median_2_2019_air_temp_max','median_2_2019_air_temp_min','median_2_2019_relative_humidity','median_2_2019_dew_point_avg','median_2_2019_dew_point_min','median_2_2019_eto_result']
fresh_weight_col = pd.DataFrame(columns=['feat', 'colname'])

fresh_weight_col['feat'] = features_fresh_weight
fresh_weight_col['colname']='0'
for i in range(len(column_names)):
    fresh_weight_col['colname'].iloc[i] = column_names[i]



fresh_weight_col = fresh_weight_col[['colname','feat']]

pd.set_option('display.max_rows', 100)
print(fresh_weight_col.sort_values(by=['feat'], ascending=False))


# In[ ]:





# # Gradient Boosting Regressor

# In[80]:


clf = MultiOutputRegressor(GradientBoostingRegressor(max_depth=5, max_features= 'auto',n_estimators = 100,min_samples_leaf = 3,min_samples_split = 10,random_state=0))
clf.fit(X_train, y_train)


# In[81]:


clf.score(X_test, y_test, sample_weight=None)


# In[82]:


cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=5)
n_scores = cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
n_scores = absolute(n_scores)

print('MSE: %.3f' % (mean(n_scores)))


# In[83]:


ypred = clf.predict(X_test)
print(ypred)


# In[84]:


clf_grid = MultiOutputRegressor(GradientBoostingRegressor())


# In[85]:


random_grid = {'estimator__n_estimators':[100,150,200],
               'estimator__max_features': ['sqrt','auto'],
               'estimator__max_depth': [3,4,5]}


# In[86]:


xgb_grid = GridSearchCV(clf_grid,  random_grid, cv = 3)

xgb_grid = xgb_grid.fit(X_train, y_train)


# In[87]:


xgb_grid.best_params_


# In[88]:


xgb_grid.score(X_test, y_test)


# In[89]:


cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=5)
n_scores = cross_val_score(xgb_grid, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
n_scores = absolute(n_scores)

print('MSE: %.3f' % (mean(n_scores)))


# In[90]:


# predict on test data

ypred_rf = clf.predict(X_test)
ypred_rf = pd.DataFrame(ypred_rf)
print(ypred_rf)


ypred_rf = ypred_rf.reset_index(drop=True)
ypred_rf.to_csv('ypred_rf.csv')
X_test_withdate = X_test_withdate.reset_index(drop=True)
X_test_withdate.to_csv('X_test_withdate.csv')

# merge the predictions with input variables

rf_merged = pd.concat([X_test_withdate,ypred_rf],axis=1,ignore_index=True)


rf_merged = rf_merged.rename(columns={0:'batch_number',	1:'class',	2:'leaf_area',	3:'Region',	4:'flight_plant_date_diff',	5:'check_plant_date_diff',	6:'check_flight_date_diff',	7:'check_date',	8:'plant_date_final',	9:'flight_date_final',	10:'mean_solar_radiation',	11:'mean_precipitation',	12:'mean_wind_speed_avg',	13:'mean_wind_speed_max',	14:'mean_battery_voltage',	15:'mean_leaf_wetness',	16:'mean_air_temp_avg',	17:'mean_air_temp_max',	18:'mean_air_temp_min',	19:'mean_relative_humidity',	20:'mean_dew_point_avg',	21:'mean_dew_point_min',	22:'mean_eto_result',	23:'median_solar_radiation',	24:'median_precipitation',	25:'median_wind_speed_avg',	26:'median_wind_speed_max',	27:'median_battery_voltage',	28:'median_leaf_wetness',	29:'median_air_temp_avg',	30:'median_air_temp_max',	31:'median_air_temp_min',	32:'median_relative_humidity',	33:'median_dew_point_avg',	34:'median_dew_point_min',	35:'median_eto_result',	36:'mean_2019_solar_radiation',	37:'mean_2019_precipitation',	38:'mean_2019_wind_speed_avg',	39:'mean_2019_wind_speed_max',	40:'mean_2019_battery_voltage',	41:'mean_2019_leaf_wetness',	42:'mean_2019_air_temp_avg',	43:'mean_2019_air_temp_max',	44:'mean_2019_air_temp_min',	45:'mean_2019_relative_humidity',	46:'mean_2019_dew_point_avg',	47:'mean_2019_dew_point_min',	48:'mean_2019_eto_result',	49:'median_2019_solar_radiation',	50:'median_2019_precipitation',	51:'median_2019_wind_speed_avg',	52:'median_2019_wind_speed_max',	53:'median_2019_battery_voltage',	54:'median_2019_leaf_wetness',	55:'median_2019_air_temp_avg',	56:'median_2019_air_temp_max',	57:'median_2019_air_temp_min',	58:'median_2019_relative_humidity',	59:'median_2019_dew_point_avg',	60:'median_2019_dew_point_min',	61:'median_2019_eto_result',	62:'mean_2_2019_solar_radiation',	63:'mean_2_2019_precipitation',	64:'mean_2_2019_wind_speed_avg',	65:'mean_2_2019_wind_speed_max',	66:'mean_2_2019_battery_voltage',	67:'mean_2_2019_leaf_wetness',	68:'mean_2_2019_air_temp_avg',	69:'mean_2_2019_air_temp_max',	70:'mean_2_2019_air_temp_min',	71:'mean_2_2019_relative_humidity',	72:'mean_2_2019_dew_point_avg',	73:'mean_2_2019_dew_point_min',	74:'mean_2_2019_eto_result',	75:'median_2_2019_solar_radiation',	76:'median_2_2019_precipitation',	77:'median_2_2019_wind_speed_avg',	78:'median_2_2019_wind_speed_max',	79:'median_2_2019_battery_voltage',	80:'median_2_2019_leaf_wetness',	81:'median_2_2019_air_temp_avg',	82:'median_2_2019_air_temp_max',	83:'median_2_2019_air_temp_min',	84:'median_2_2019_relative_humidity',	85:'median_2_2019_dew_point_avg',	86:'median_2_2019_dew_point_min',	87:'median_2_2019_eto_result',
                                      88:'fresh_weight',	89:'radial_diameter',	90:'polar_diameter'
})
print(rf_merged)
rf_merged.to_csv('rf_merged.csv')


# In[91]:


# Analyse the predictions

rf_merged_1 = rf_merged[['check_date','fresh_weight']]
rf_merged_2 = rf_merged_1.groupby(['check_date'],sort=True).sum()
rf_merged_2 = rf_merged_2.sort_values(by=['check_date'])


rf_merged_1 = rf_merged[['check_date']]
rf_merged_1['check_date_copy'] = rf_merged_1['check_date']
rf_merged_3 = rf_merged_1.groupby(['check_date_copy'],sort=True).max()
rf_merged_3 = rf_merged_3.sort_values(by=['check_date'])


rf_merged_2 = rf_merged_2.reset_index(drop=True)
rf_merged_3 = rf_merged_3.reset_index(drop=True)

rf_merged_4 = pd.concat([rf_merged_2,rf_merged_3],axis=1,ignore_index=True)
rf_merged_4 = rf_merged_4.rename(columns={0:'sum_fresh_weight',	1:'check_date'})
rf_merged_4 = rf_merged_4[['check_date','sum_fresh_weight']]
rf_merged_4


# In[101]:


# plot of expected lettuce produce over coming days

plt.plot(rf_merged_4['check_date'],rf_merged_4['sum_fresh_weight'])
plt.rcParams["figure.figsize"] = [100,5]
plt.title('Lettuce Produce over Days')
plt.xlabel('Date')
plt.ylabel('Weight of Produce')
plt.show()


# In[93]:


# Which region has highest weight of lettuce ?

rf_merged_1 = rf_merged[['Region','fresh_weight']]
rf_merged_2 = rf_merged_1.groupby(['Region'],sort=True).sum()
rf_merged_2

# Since the test data has only one single region, we cant compare weight of produce across different regions.


# In[94]:


# how many plants were planted in each region ?

rf_merged_1 = rf_merged[['Region','fresh_weight']]
rf_merged_2 = rf_merged_1.groupby(['Region'],sort=True).count()
rf_merged_2

# we can take average produce for each region (which is total produce divided by number of plants) and see which region has 
# highest weight per plant. More plants can be planted here as the produce is better than other regions.Weather conditions of 
# this region can be suitable for plants which is resulting in better produce


# In[95]:


dummy_regressor = DummyRegressor(strategy="mean")
dummy_regressor.fit(X_train, y_train)


# In[96]:


dummy_regressor.score(X_train, y_train)


# In[97]:


# Check feature importances for fresh weight

features_fresh_weight = clf.estimators_[0].feature_importances_
column_names =['batch_number','class','leaf_area','Region','flight_plant_date_diff','check_plant_date_diff','check_flight_date_diff','mean_solar_radiation','mean_precipitation','mean_wind_speed_avg','mean_wind_speed_max','mean_battery_voltage','mean_leaf_wetness','mean_air_temp_avg','mean_air_temp_max','mean_air_temp_min','mean_relative_humidity','mean_dew_point_avg','mean_dew_point_min','mean_eto_result','median_solar_radiation','median_precipitation','median_wind_speed_avg','median_wind_speed_max','median_battery_voltage','median_leaf_wetness','median_air_temp_avg','median_air_temp_max','median_air_temp_min','median_relative_humidity','median_dew_point_avg','median_dew_point_min','median_eto_result','mean_2019_solar_radiation','mean_2019_precipitation','mean_2019_wind_speed_avg','mean_2019_wind_speed_max','mean_2019_battery_voltage','mean_2019_leaf_wetness','mean_2019_air_temp_avg','mean_2019_air_temp_max','mean_2019_air_temp_min','mean_2019_relative_humidity','mean_2019_dew_point_avg','mean_2019_dew_point_min','mean_2019_eto_result','median_2019_solar_radiation','median_2019_precipitation','median_2019_wind_speed_avg','median_2019_wind_speed_max','median_2019_battery_voltage','median_2019_leaf_wetness','median_2019_air_temp_avg','median_2019_air_temp_max','median_2019_air_temp_min','median_2019_relative_humidity','median_2019_dew_point_avg','median_2019_dew_point_min','median_2019_eto_result','mean_2_2019_solar_radiation','mean_2_2019_precipitation','mean_2_2019_wind_speed_avg','mean_2_2019_wind_speed_max','mean_2_2019_battery_voltage','mean_2_2019_leaf_wetness','mean_2_2019_air_temp_avg','mean_2_2019_air_temp_max','mean_2_2019_air_temp_min','mean_2_2019_relative_humidity','mean_2_2019_dew_point_avg','mean_2_2019_dew_point_min','mean_2_2019_eto_result','median_2_2019_solar_radiation','median_2_2019_precipitation','median_2_2019_wind_speed_avg','median_2_2019_wind_speed_max','median_2_2019_battery_voltage','median_2_2019_leaf_wetness','median_2_2019_air_temp_avg','median_2_2019_air_temp_max','median_2_2019_air_temp_min','median_2_2019_relative_humidity','median_2_2019_dew_point_avg','median_2_2019_dew_point_min','median_2_2019_eto_result']
fresh_weight_col = pd.DataFrame(columns=['feat', 'colname'])

fresh_weight_col['feat'] = features_fresh_weight
fresh_weight_col['colname']='0'
for i in range(len(column_names)):
    fresh_weight_col['colname'].iloc[i] = column_names[i]



fresh_weight_col = fresh_weight_col[['colname','feat']]

pd.set_option('display.max_rows', 100)
print(fresh_weight_col.sort_values(by=['feat'], ascending=False))


# In[98]:


# Check feature importances for radial diameter

features_fresh_weight = clf.estimators_[1].feature_importances_
column_names =['batch_number','class','leaf_area','Region','flight_plant_date_diff','check_plant_date_diff','check_flight_date_diff','mean_solar_radiation','mean_precipitation','mean_wind_speed_avg','mean_wind_speed_max','mean_battery_voltage','mean_leaf_wetness','mean_air_temp_avg','mean_air_temp_max','mean_air_temp_min','mean_relative_humidity','mean_dew_point_avg','mean_dew_point_min','mean_eto_result','median_solar_radiation','median_precipitation','median_wind_speed_avg','median_wind_speed_max','median_battery_voltage','median_leaf_wetness','median_air_temp_avg','median_air_temp_max','median_air_temp_min','median_relative_humidity','median_dew_point_avg','median_dew_point_min','median_eto_result','mean_2019_solar_radiation','mean_2019_precipitation','mean_2019_wind_speed_avg','mean_2019_wind_speed_max','mean_2019_battery_voltage','mean_2019_leaf_wetness','mean_2019_air_temp_avg','mean_2019_air_temp_max','mean_2019_air_temp_min','mean_2019_relative_humidity','mean_2019_dew_point_avg','mean_2019_dew_point_min','mean_2019_eto_result','median_2019_solar_radiation','median_2019_precipitation','median_2019_wind_speed_avg','median_2019_wind_speed_max','median_2019_battery_voltage','median_2019_leaf_wetness','median_2019_air_temp_avg','median_2019_air_temp_max','median_2019_air_temp_min','median_2019_relative_humidity','median_2019_dew_point_avg','median_2019_dew_point_min','median_2019_eto_result','mean_2_2019_solar_radiation','mean_2_2019_precipitation','mean_2_2019_wind_speed_avg','mean_2_2019_wind_speed_max','mean_2_2019_battery_voltage','mean_2_2019_leaf_wetness','mean_2_2019_air_temp_avg','mean_2_2019_air_temp_max','mean_2_2019_air_temp_min','mean_2_2019_relative_humidity','mean_2_2019_dew_point_avg','mean_2_2019_dew_point_min','mean_2_2019_eto_result','median_2_2019_solar_radiation','median_2_2019_precipitation','median_2_2019_wind_speed_avg','median_2_2019_wind_speed_max','median_2_2019_battery_voltage','median_2_2019_leaf_wetness','median_2_2019_air_temp_avg','median_2_2019_air_temp_max','median_2_2019_air_temp_min','median_2_2019_relative_humidity','median_2_2019_dew_point_avg','median_2_2019_dew_point_min','median_2_2019_eto_result']
fresh_weight_col = pd.DataFrame(columns=['feat', 'colname'])

fresh_weight_col['feat'] = features_fresh_weight
fresh_weight_col['colname']='0'
for i in range(len(column_names)):
    fresh_weight_col['colname'].iloc[i] = column_names[i]



fresh_weight_col = fresh_weight_col[['colname','feat']]

pd.set_option('display.max_rows', 100)
print(fresh_weight_col.sort_values(by=['feat'], ascending=False))


# In[99]:


# Check feature importances for polar diameter

features_fresh_weight = clf.estimators_[2].feature_importances_
column_names =['batch_number','class','leaf_area','Region','flight_plant_date_diff','check_plant_date_diff','check_flight_date_diff','mean_solar_radiation','mean_precipitation','mean_wind_speed_avg','mean_wind_speed_max','mean_battery_voltage','mean_leaf_wetness','mean_air_temp_avg','mean_air_temp_max','mean_air_temp_min','mean_relative_humidity','mean_dew_point_avg','mean_dew_point_min','mean_eto_result','median_solar_radiation','median_precipitation','median_wind_speed_avg','median_wind_speed_max','median_battery_voltage','median_leaf_wetness','median_air_temp_avg','median_air_temp_max','median_air_temp_min','median_relative_humidity','median_dew_point_avg','median_dew_point_min','median_eto_result','mean_2019_solar_radiation','mean_2019_precipitation','mean_2019_wind_speed_avg','mean_2019_wind_speed_max','mean_2019_battery_voltage','mean_2019_leaf_wetness','mean_2019_air_temp_avg','mean_2019_air_temp_max','mean_2019_air_temp_min','mean_2019_relative_humidity','mean_2019_dew_point_avg','mean_2019_dew_point_min','mean_2019_eto_result','median_2019_solar_radiation','median_2019_precipitation','median_2019_wind_speed_avg','median_2019_wind_speed_max','median_2019_battery_voltage','median_2019_leaf_wetness','median_2019_air_temp_avg','median_2019_air_temp_max','median_2019_air_temp_min','median_2019_relative_humidity','median_2019_dew_point_avg','median_2019_dew_point_min','median_2019_eto_result','mean_2_2019_solar_radiation','mean_2_2019_precipitation','mean_2_2019_wind_speed_avg','mean_2_2019_wind_speed_max','mean_2_2019_battery_voltage','mean_2_2019_leaf_wetness','mean_2_2019_air_temp_avg','mean_2_2019_air_temp_max','mean_2_2019_air_temp_min','mean_2_2019_relative_humidity','mean_2_2019_dew_point_avg','mean_2_2019_dew_point_min','mean_2_2019_eto_result','median_2_2019_solar_radiation','median_2_2019_precipitation','median_2_2019_wind_speed_avg','median_2_2019_wind_speed_max','median_2_2019_battery_voltage','median_2_2019_leaf_wetness','median_2_2019_air_temp_avg','median_2_2019_air_temp_max','median_2_2019_air_temp_min','median_2_2019_relative_humidity','median_2_2019_dew_point_avg','median_2_2019_dew_point_min','median_2_2019_eto_result']
fresh_weight_col = pd.DataFrame(columns=['feat', 'colname'])

fresh_weight_col['feat'] = features_fresh_weight
fresh_weight_col['colname']='0'
for i in range(len(column_names)):
    fresh_weight_col['colname'].iloc[i] = column_names[i]



fresh_weight_col = fresh_weight_col[['colname','feat']]

pd.set_option('display.max_rows', 100)
print(fresh_weight_col.sort_values(by=['feat'], ascending=False))


# In[ ]:




