#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns


# In[10]:


cd /Users/Owner/Downloads


# In[11]:


pwd


# In[12]:


raw_data = pd.read_csv('oura_data.csv')


# In[13]:


raw_data.head()


# In[14]:


raw_data.shape


# In[15]:


raw_data.columns


# In[16]:


'''
I deleted all activity related variables since I have not tracked workouts and thus is inaccurate
This was the first command but I forgot to delete a couple more thus I had to run a new command
I leave this as a comment to be able to return to it if needed later.

raw_data.drop(['Activity Score',
       'Stay Active Score', 'Move Every Hour Score',
       'Meet Daily Targets Score', 'Training Frequency Score',
       'Training Volume Score', 'Recovery Time Score', 'Activity Burn',
       'Total Burn', 'Target Calories', 'Steps', 'Daily Movement',
       'Inactive Time', 'Rest Time', 'Low Activity Time',
       'Medium Activity Time', 'High Activity Time', 'Non-wear Time',
       'Average MET', 'Long Periods of Inactivity'],inplace=True,axis=1)
       
'''

raw_data.drop(['Previous Day Activity Score', 'Activity Balance Score'], inplace=True, axis=1)


# In[17]:


raw_data.columns


# In[18]:


sns.set(rc={'figure.figsize':(12,10)})
daily_line = sns.lineplot(x='date', y='Readiness Score', data=raw_data)
daily_line.set_title('Daily Readiness')


# In[20]:


raw_data['date'] = pd.to_datetime(raw_data['date'])
week_data = raw_data.resample('W-Mon', on='date').mean().reset_index().sort_values(by='date')


# In[51]:


week_line = sns.lineplot(x='date', y='Readiness Score', data=week_data)
week_line.set_title('Weekly Readiness Average')


# In[21]:


# we must replace nan values with 0s since these were all nighters
raw_data = raw_data.fillna(0)


# In[22]:


total_sleep_score = raw_data['Total Sleep Score'].values
sns.distplot(total_sleep_score, color = '#bd3999').set_title('Total Sleep Score Histogram')


# In[23]:


total_sleep_time = raw_data['Total Sleep Time'].values
sns.distplot(total_sleep_time, color = '#bd3999').set_title('Total Sleep Time Histogram')


# In[24]:


readiness_score = raw_data['Readiness Score'].values
sns.distplot(readiness_score, color = '#bd3999').set_title('Readiness Score Histogram')


# In[82]:


# Creating pairplots to see scatterplots and distributions
pairplot_1 = sns.pairplot(raw_data[['Readiness Score', 'REM Sleep Score', 'Total Sleep Time', 'Average Resting Heart Rate']], 
                 plot_kws={'color':'#328ba8'})


# In[81]:


pairplot_2 = sns.pairplot(raw_data[['Lowest Resting Heart Rate', 'Average HRV', 'Temperature Deviation (°C)', 'Readiness Score']],
                         plot_kws={'color':'#eb7fdc'})


# In[25]:


raw_data.columns


# # Correlation Matrix and Correlation Heat Map

# In[185]:


#Correlation Matrix
raw_data.corr()


# In[190]:


cmap = sns.diverging_palette(220,10,as_cmap=True)
sns.heatmap(raw_data.corr(),vmax=.3, center=0, cmap=cmap,
           square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[26]:


attributes = ['Total Bedtime', 'Total Sleep Time', 'Awake Time', 'REM Sleep Time', 'Light Sleep Time',
                     'Deep Sleep Time', 'Restless Sleep', 'Sleep Efficiency', 'Sleep Latency', 'Average Resting Heart Rate',
                     'Lowest Resting Heart Rate', 'Average HRV', 'Temperature Deviation (°C)', 'Respiratory Rate',
                     'Readiness Score']
cmap = sns.diverging_palette(220,10,as_cmap=True)
sns.heatmap(raw_data[attributes].corr(),vmax=.3, center=0, cmap=cmap,
           square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[166]:


# now we split raw_data into train and test sets
train_dataset = raw_data.sample(frac=0.8, random_state=0)
test_dataset = raw_data.drop(train_dataset.index)


# In[167]:


train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Readiness Score')
test_labels = test_features.pop('Readiness Score')


# # PCA Regression

# In[85]:


# first we extract the principal components that account for 95% of the variance
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(train_features)

train_img = scaler.transform(train_features)
test_img = scaler.transform(test_features)


# In[86]:


from sklearn.decomposition import PCA

pca = PCA(.95)
pca.fit(train_img)

train_img = pca.transform(train_img)
test_img = pca.transform(test_img)


# In[87]:


from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(train_img, train_labels)


# In[99]:


from sklearn.metrics import mean_squared_error

y_pred = linear_regression.predict(test_img)
y_true = test_labels.values

mean_squared_error(y_true, y_pred)


# # Gradient Boosting Regression

# In[118]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels)

gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)

min_val_error = float('inf')
error_going_up = 0
for n_estimators in range(1,120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up +=1
        if error_going_up == 5:
            break


# In[120]:


y_pred = gbrt.predict(test_features)

mean_squared_error(y_true, y_pred)


# In[165]:


train_features.shape


# # Regular MLP

# In[121]:


import tensorflow as tf


# In[139]:


from keras.layers import Dense, Activation
from keras.models import Sequential

sc = StandardScaler()
X_train = sc.fit_transform(train_features)
X_test = sc.transform(test_features)


model = Sequential([
    Dense(50, activation='relu', input_shape=X_train.shape[1:]),
    Dense(40, activation='relu'),
    Dense(30, activation='relu'),
    Dense(20, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='relu')
    
])

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, train_labels, batch_size = 10, epochs = 100)


# In[141]:


y_pred = model.predict(X_test)

mean_squared_error(y_true, y_pred)


# # Wide & Deep NN

# In[155]:


input_ = keras.layers.Input(shape = X_train.shape[1:])
hidden1 = keras.layers.Dense(50, activation='relu')(input_)
hidden2 = keras.layers.Dense(40, activation='relu')(hidden1)
hidden3 = keras.layers.Dense(30, activation='relu')(hidden2)
hidden4 = keras.layers.Dense(20, activation='relu')(hidden3)
hidden5 = keras.layers.Dense(10, activation='relu')(hidden4)
concat = keras.layers.Concatenate()([input_, hidden5])
output = keras.layers.Dense(1, activation='relu')(concat)
wide_model=keras.Model(inputs=[input_], outputs=[output])

wide_model.compile(loss='mean_squared_error', optimizer='adam')

wide_model.fit(X_train, train_labels, batch_size = 10, epochs = 100)


# In[156]:


y_pred = wide_model(X_test)

mean_squared_error(y_true, y_pred)


# # Conclusion
# ## The best performing model (under MSE) was the PCA Regressor
# ## Although the neural networks underperformed the other two models this can be due
# ## to the size of the dataset.

# In[171]:


pwd

