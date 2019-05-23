# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:57:48 2019

@author: macwanfr
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats

# Importing the dataset
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

dataset_train.shape
dataset_test.shape

dataset_train.head()
dataset_test.head()

dataset_train.describe()
dataset_test.describe()

non_obj = [f for f in dataset_train.columns if dataset_train.dtypes[f] != 'object']
non_obj.remove('SalePrice')
non_obj.remove('Id')
obj = [f for f in dataset_train.columns if dataset_train.dtypes[f] == 'object']


#sns.set_style("whitegrid")
missing = dataset_train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()

y = dataset_train['SalePrice']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=stats.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=stats.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=stats.lognorm)



def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()
    
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature+'_Encode'] = o
    
qual_encoded = []
for q in obj:  
    encode(dataset_train, q)
    qual_encoded.append(q+'_Encode')
print(qual_encoded)



plt.figure(1)
corr = dataset_train[non_obj+['SalePrice']].corr()
sns.heatmap(corr)
plt.figure(2)
corr = dataset_train[qual_encoded+['SalePrice']].corr()
sns.heatmap(corr)
plt.figure(3)
corr = pd.DataFrame(np.zeros([len(non_obj)+1, len(qual_encoded)+1]), index=non_obj+['SalePrice'], columns=qual_encoded+['SalePrice'])
for q1 in non_obj+['SalePrice']:
    for q2 in qual_encoded+['SalePrice']:
        corr.loc[q1, q2] = dataset_train[q1].corr(dataset_train[q2])
sns.heatmap(corr)


#data processing

dataset_train.drop(['Id'], axis=1, inplace=True)
dataset_test.drop(['Id'], axis=1, inplace=True)
dataset_train = dataset_train[dataset_train.GrLivArea < 4500]
dataset_train.reset_index(drop=True, inplace=True)
dataset_train["SalePrice"] = np.log1p(dataset_train["SalePrice"])
y = dataset_train['SalePrice'].reset_index(drop=True)

train_cols = dataset_train.drop(['SalePrice'], axis=1)
test_cols = dataset_test

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    train_cols[col] = train_cols[col].fillna(0)
    test_cols[col] = test_cols[col].fillna(0)
    
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train_cols[col] = train_cols[col].fillna('None')
    test_cols[col] = test_cols[col].fillna('None')
    
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train_cols[col] = train_cols[col].fillna('None')
    test_cols[col] = test_cols[col].fillna('None')
    
train_cols['MSZoning'] = train_cols.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
test_cols['MSZoning'] = test_cols.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

train_cols.head()

test_cols.head()

X = dataset_train.fillna(0.).values

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

std = StandardScaler()
s = std.fit_transform(X)
pca = PCA(n_components=30)
pca.fit(s)
pc = pca.transform(s)

a=list(dataset_train)

dataset_train.shape

dataset_train['MSZoning'].describe()

dataset_train.dtypes

X_train = dataset_train.iloc[:, :-1].values
y_train = dataset_train.iloc[:, -1].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
dataset_train['MSZoning'] = labelencoder.fit_transform(dataset_train['MSZoning'])
onehotencoder = OneHotEncoder(categorical_features = [5])
dataset_train['MSZoning'] = onehotencoder.fit_transform(dataset_train['MSZoning']).toarray()

dataset_test = pd.read_csv('test.csv')
X_test = dataset_test.iloc[:, :].values
#y_test = dataset_test.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)