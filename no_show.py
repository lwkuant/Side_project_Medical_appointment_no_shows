# -*- coding: utf-8 -*-
"""
Medical Appointment No Shows
https://www.kaggle.com/joniarroba/noshowappointments
"""


### setup

import os 
os.chdir(r'D:\Dataset\Side_project_Medical_appointment_no_show')

## import required packages
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns

## set random seed
seed = 100
np.random.seed(seed)

## load the dataset
df = pd.read_csv('No-show-Issue-Comma-300k.csv')

## Overview 
print(df.shape)

print(df.head())

print(df.info())

print(df.isnull().any()) # for NAs
# No NAs

print(df['Status'].value_counts()) # for the target ratio
print(df['Status'].value_counts().apply(lambda x: x/np.sum(df['Status'].value_counts().values)*100)) # ratio in percentage
# roughly 2-to-1

## transform the target variable into value for the model 
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Status_val'] = encoder.fit_transform(df['Status'])
print(df['Status_val'].value_counts())

# I want the 'No-Show' to be 1 instead
label_map = {'Show-Up':0, 'No-Show':1}
df['Status_val'] = df['Status'].map(label_map)
print(df['Status_val'].value_counts())
# well-done
df.drop(['Status'], axis=1, inplace=True)

# get the test data for further testing 
from sklearn.cross_validation import train_test_split
X_tr, X_test, y_tr, y_test = train_test_split(df.ix[:,  df.columns != 'Status_val'], df['Status_val'], test_size=0.2, stratify=df['Status_val'], random_state=seed)
df_tr = pd.concat([X_tr, y_tr], axis=1)
df_tr.index = range(len(df_tr))
print(df_tr.head())


### EDA

## Age to Status

plt.figure(figsize=[15, 15])
for status in np.unique(df_tr['Status_val'].values):
    sns.distplot(df_tr['Age'].values[df_tr['Status_val'].values == status],
                 kde_kws={'alpha':1, 'label': status, 'lw':3})
    plt.legend()
# most of the two plots are overlapped, however there are still some trends
# no_show with higher probability of the younger
# show_up with higher probability of the older

plt.figure(figsize=[15, 15])
for status in np.unique(df_tr['Status_val'].values):
    sns.distplot(df_tr['Age'].values[df_tr['Status_val'].values == status],
                 hist_kws={'label': str(status)}, kde=False)
    plt.legend()
# at certain numbers, the count would be higher for both statuses
    
# I also observe that there are some ages are negative
print(np.sum(df_tr['Age']<0))
# remove them 
df_tr = df_tr.ix[df_tr['Age'] >= 0, :]
df_tr.index = range(len(df_tr))

## Gender to Status

gender_cross_tb = pd.crosstab(df_tr['Gender'], df_tr['Status_val'])
print(gender_cross_tb)
print(gender_cross_tb.apply(lambda x: x/np.sum(gender_cross_tb.values, axis=1)))
gender_cross_tb.apply(lambda x: x/np.sum(gender_cross_tb.values, axis=1)).plot(kind='bar', stacked=True, colormap='Paired', edgecolor='none', rot=0)
plt.subplots_adjust(bottom=0.15, right=0.8)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
# from the outcomes of the cross table and bar plot, we can find out that the sex doesn't 
# serve as a good predictor, since the ratio doesn't show significant difference

## DayOfTheWeek

week_cross_tb = pd.crosstab(df_tr['DayOfTheWeek'], df_tr['Status_val'])
print(week_cross_tb)
print(week_cross_tb.apply(lambda x: x/np.sum(week_cross_tb.values, axis=1)))
week_cross_tb.apply(lambda x: x/np.sum(week_cross_tb.values, axis=1)).plot(kind='bar', stacked=True, colormap='Paired', edgecolor='none', rot=0)
plt.subplots_adjust(bottom=0.15, right=0.8)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

# combine Saturday and Sunday as one group, the rest also becomes one group
def week_map(week):
    if week in ['Saturday', 'Sunday']:
        return 1
    else:
        return 0

df_tr['Voc'] = df_tr['DayOfTheWeek'].map(week_map)
df_tr['Voc'].value_counts()
# test the result 
week_cross_tb = pd.crosstab(df_tr['Voc'], df_tr['Status_val'])
print(week_cross_tb)
print(week_cross_tb.apply(lambda x: x/np.sum(week_cross_tb.values, axis=1)))
week_cross_tb.apply(lambda x: x/np.sum(week_cross_tb.values, axis=1)).plot(kind='bar', stacked=True, colormap='Paired', edgecolor='none', rot=0)
plt.subplots_adjust(bottom=0.15, right=0.8)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

## the rest of the categorical features

cat_features = ['Diabetes', 'Alcoolism', 'HiperTension', 'Handcap',
       'Smokes', 'Scholarship', 'Tuberculosis', 'Sms_Reminder']

# test if they all only have two categories
for feature in cat_features:
    print(feature, len(np.unique(df_tr[feature])))

# more than two categories
print(df_tr['Handcap'].value_counts())    
print(df_tr['Sms_Reminder'].value_counts())    
    
chi_dict = {}

from scipy.stats import chi2_contingency

for feature in cat_features:
    cross_tab = pd.crosstab(df_tr[feature], df_tr['Status_val'])
    print(feature)
    print(cross_tab)
    print(cross_tab.apply(lambda x: x/np.sum(cross_tab.values, axis=1)))
    
    cross_tab.apply(lambda x: x/np.sum(cross_tab.values, axis=1)).plot(kind='bar', stacked=True, colormap='Paired', edgecolor='none', rot=0)
    plt.subplots_adjust(bottom=0.15, right=0.8)
    plt.title(feature)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    chi_dict[feature] = chi2_contingency(cross_tab)[:2]

print(chi_dict)
print(sorted(chi_dict, key=chi_dict.get, reverse=True))

# keep: ['HiperTension', 'Scholarship', 'Diabetes', 'Smokes', 'Alcoolism', 'Handcap']

## determine which categorical features to keep
# test if the following features are correlated
# ['Voc', HiperTension', 'Scholarship', 'Diabetes', 'Smokes', 'Alcoolism', 'Handcap']
cat_features = ['Voc', 'HiperTension', 'Scholarship', 'Diabetes', 'Smokes', 'Alcoolism', 'Handcap']
chi_table = pd.DataFrame(columns = cat_features, index = cat_features)

for feature_1 in cat_features:
    for feature_2 in cat_features:
        if feature_1 == feature_2:
            chi_table.ix[feature_1, feature_2] = 1
        else:
            chi_table.ix[feature_1, feature_2] = \
                chi2_contingency(pd.crosstab(df_tr[feature_1], df_tr[feature_2]))[1]

sns.heatmap(chi_table.values.astype(float), annot=True, xticklabels=cat_features, yticklabels=cat_features)             
plt.xticks(rotation=45)

chi_dict = {}

for feature in cat_features:
    cross_tab = pd.crosstab(df_tr[feature], df_tr['Status_val'])
    chi_dict[feature] = chi2_contingency(cross_tab)[:2]

print(chi_dict)
print(sorted(chi_dict, key=chi_dict.get, reverse=True))

# features to keep: ['HiperTension', 'Scholarshiip', 'Voc']

## AwaitingTime
plt.figure(figsize=[15, 15])
for status in np.unique(df_tr['Status_val'].values):
    sns.distplot(df_tr['AwaitingTime'].values[df_tr['Status_val'].values == status],
                 kde_kws={'alpha':1, 'label': status, 'lw':3})
    plt.legend()

plt.figure(figsize=[15, 15])
for status in np.unique(df_tr['Status_val'].values):
    sns.distplot(df_tr['AwaitingTime'].values[df_tr['Status_val'].values == status],
                 hist_kws={'label': str(status)}, kde=False)
    plt.legend()
    
# transform the AwaitingTime to absolute value
df_tr['AwaitingTime_pos'] = df_tr['AwaitingTime']*-1

plt.figure(figsize=[15, 15])
for status in np.unique(df_tr['Status_val'].values):
    sns.distplot(df_tr['AwaitingTime_pos'].values[df_tr['Status_val'].values == status],
                 kde_kws={'alpha':1, 'label': status, 'lw':3})
    plt.legend()
# right-skewed

# plot the log-transformed values
df_tr['AwaitingTime_pos_trans'] = np.log1p(df_tr['AwaitingTime_pos'])

plt.figure(figsize=[15, 15])
for status in np.unique(df_tr['Status_val'].values):
    sns.distplot(df_tr['AwaitingTime_pos_trans'].values[df_tr['Status_val'].values == status],
                 kde_kws={'alpha':1, 'label': status, 'lw':3})
    plt.legend()

plt.figure(figsize=[15, 15])
for status in np.unique(df_tr['Status_val'].values):
    sns.distplot(df_tr['AwaitingTime_pos_trans'].values[df_tr['Status_val'].values == status],
                 hist_kws={'label': str(status)}, kde=False)
    plt.legend()
# AwaitingTime is not a good feature, since the distributions are similar for both statuses 

## combine Alcoholism with Diatbetes
df_tr['try'] = df_tr['Alcoolism']*df_tr['Smokes']

# test the correlation with target
print(chi2_contingency(pd.crosstab(df_tr['try'], df_tr['Status_val'])))

# test the correlation with the selected features
cat_features = ['Voc', 'HiperTension', 'Scholarship', 'try']
chi_table = pd.DataFrame(columns = cat_features, index = cat_features)

for feature_1 in cat_features:
    for feature_2 in cat_features:
        if feature_1 == feature_2:
            chi_table.ix[feature_1, feature_2] = 1
        else:
            chi_table.ix[feature_1, feature_2] = \
                chi2_contingency(pd.crosstab(df_tr[feature_1], df_tr[feature_2]))[1]

sns.heatmap(chi_table.values.astype(float), annot=True, xticklabels=cat_features, yticklabels=cat_features)             
plt.xticks(rotation=45)   
# keep the new feature
df_tr['Al_Sm'] = df_tr['Alcoolism']*df_tr['Smokes']
    

### Modeling 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

X = df_tr.ix[:,['Age', 'Voc', 'HiperTension', 'Scholarship', 'Al_Sm']].values
y = df_tr['Status_val'].values

np.random.seed(seed)
model = RandomForestClassifier()
score = cross_val_score(model, X, y, cv=5, scoring='precision', n_jobs=-1, verbose=1, )
print(score)
# not very good, since the precision is under 0.5



### Evaluate on test data
# not yet done
def transform_df(df):
    
    # remove the age that is negative 
    df = df.ix[df['Age'] >= 0, :]
    df.index = range(len(df))
    
    # construct new feature: Voc
    def week_map(week):
    if week in ['Saturday', 'Sunday']:
        return 1
    else:
        return 0
        
    df['Voc'] = df['DayOfTheWeek'].map(week_map)
    
    #
    
    # columns to keep: ['Age', 'Voc', 'HiperTension', 'Scholarship', 'Al_Sm']    
    




