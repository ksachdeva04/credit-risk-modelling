import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import warnings
import os

a1=pd.read_csv('data/case_study1.csv')
a2=pd.read_csv('data/case_study2.csv')

df1=a1.copy()
df2=a2.copy()
# print(df1.info())

# removing nulls
df1=df1.loc[df1['Age_Oldest_TL']!=-99999]


# check if a col has more than 20% missing values 
cols_to_be_removed=[]
for i in df2.columns:
    if df2.loc[df2[i]==-99999].shape[0]>10000:
        cols_to_be_removed.append(i)
df2=df2.drop(cols_to_be_removed,axis=1)
#removing rows with null values
for i in df2.columns:
    df2=df2.loc[df2[i]!=-99999]



# common col in df1 and df2

for i in df1.columns:
    if i in df2.columns:
        print(i)

# combining df1 and df2
df=pd.merge(df1,df2,how='inner',left_on=['PROSPECTID'],right_on=['PROSPECTID'])
print(df.head())
print(df.shape)
# divide cols into categorical and numerical
df.info()
for i in df.columns:
    if df[i].dtype=='object':
        print(i)

# chi square test

for i in ['MARITALSTATUS','EDUCATION','GENDER','last_prod_enq2','first_prod_enq2']:
    chi2,pval,_,_=chi2_contingency(pd.crosstab(df[i],df['Approved_Flag']))
    print(i,'---',pval)

# we have taken signifance level->0.05
#since all cat features have pval<=0.05,we accept them all

#numerical features
num_features=[]
for i in df.columns:
    if df[i].dtype!='object' and i not in ['PROSPECTID','Approved_Flag']:
        num_features.append(i)
print(len(num_features))

print(['MARITALSTATUS','EDUCATION','GENDER','last_prod_enq2','first_prod_enq2','PROSPECTID','Approved_Flag'] in num_features)
