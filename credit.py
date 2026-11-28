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

# Variance Inflation Factor for multicolinearity
vif_data=df[num_features]
total_columns=vif_data.shape[1]
columns_to_be_kept=[]
columns_index=0

for i in range(0,total_columns):

    vif_value=variance_inflation_factor(vif_data,columns_index)

    if vif_value<=6:
        columns_to_be_kept.append(num_features[i])
        columns_index+=1
    else:
        vif_data=vif_data.drop([num_features[i]],axis=1)
print(len(columns_to_be_kept))
print(vif_data.shape)

# our dataset is left with 46 columns 
#Multicolinearity removed

#ANOVA Test for num cols
from scipy.stats import f_oneway
cols_to_be_kept_num=[]
for i in columns_to_be_kept:
    a=list(df[i])
    b=list(df['Approved_Flag'])
    group_P1=[value for value,group in zip(a,b) if group =='P1']
    group_P2=[value for value,group in zip(a,b) if group =='P2']
    group_P3=[value for value,group in zip(a,b) if group =='P3']
    group_P4=[value for value,group in zip(a,b) if group =='P4']
    f_statistic,p_value=f_oneway(group_P1,group_P2,group_P3,group_P4)
    if p_value<=0.05:
        cols_to_be_kept_num.append(i)
print(len(cols_to_be_kept_num))