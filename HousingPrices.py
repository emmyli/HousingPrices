
# coding: utf-8

# ## Introduction

# We will be predicting the housing prices based on the given data set.

# ## Import libraries and data sets

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# ## Data exploration

# In[2]:


# Check out the first few rows of the data
train.head(10)


# In[3]:


# Quick summary of the data
train.info()


# In[4]:


# Summary of the target variable
train['SalePrice'].describe()


# ## Data visualization

# ### Target variable

# In[5]:


# Histogram of the target variable
sns.distplot(train['SalePrice']);


# We note that the histogram shows that the target variable is positively skewed.

# In[6]:


# Skewness and kurtosis of the target variable
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())


# ### Numerical and categorical variables

# We now examine some notable numerical and categorical variables and their relationships with the target variable. In particular, we will examine the GrLivArea, TotalBsmtSF, OverallQual and the YearBuilt since we believe that these variables are most likely to have a direct impact on the sales price of the house.

# #### GrLivArea: Above grade (ground) living area square feet

# In[7]:


# Scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# #### TotalBsmtSF: Total square feet of basement area

# In[8]:


# Scatter plot totalBsmtSF/saleprice
var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# #### OverallQual: Rates the overall material and finish of the house

# In[9]:


# Box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[10]:


# Box plot overallqual/saleprice
var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(20, 10))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# ### Correlation Matrix

# Since we have only looked at a couple of variabels, we will use a correlation matrix to examine the relationship between all of the variables. The table gives a quick overview of all the relationships and shows the correlation coefficient between all of the variables.

# In[11]:


# Correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[12]:


# Saleprice correlation matrix
k = 10 # number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# ### Scatterplot

# In[13]:


# Scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();


# ### Missing Data

# In[14]:


# Missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[15]:


# Dealing with the missing data
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max()


# ### Outliers

# #### Univariate Analysis

# In[16]:


# Standardizing data
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('Lower outer range of the distribution:')
print(low_range)
print('\nHigher outer range of the distribution:')
print(high_range)


# #### Bivariate Analysis

# In[17]:


# Bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[18]:


# Deleting points
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524].index)


# In[19]:


# New bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# ## Data wrangling

# ### Testing Assumptions
# **Normality** - the assumption that the data can be approximated by normal distribution <br>
# **Homoscedasticity** - the assumption that dependent variable(s) exhibit equal levels of variance across the range of predictor variable(s) <br>
# **Linearity** - the assumption of the existence of linear relationships <br>
# **Absence of correlated errors** - the assumption of the absence of correlated errors

# ### Normality

# #### SalePrice

# In[20]:


# Histogram and normal probability plot
sns.distplot(train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)


# We observe that the 'SalesPrice' is not normal since the probability plot clearly does not follow the diagonal line, and hence we will apply a log transformation in hopes of making 'SalesPrice' normal.

# In[21]:


# Applying log transformation
train['SalePrice'] = np.log(train['SalePrice'])


# In[22]:


# Transformed histogram and normal probability plot
sns.distplot(train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)


# #### GrLivArea

# In[23]:


# Histogram and normal probability plot
sns.distplot(train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)


# Similarly, we apply a log transformation since it exhibits positive skewness and does not follow normal distribution.

# In[24]:


# Data transformation
train['GrLivArea'] = np.log(train['GrLivArea'])

# Transformed histogram and normal probability plot
sns.distplot(train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)


# #### TotalBsmtSF

# In[25]:


# Histogram and normal probability plot
sns.distplot(train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'], plot=plt)


# Since there are a signicant number of 0 values for the variable, we cannot apply a log transformation. We will create a new variable 'HasBsmt' to separate the houses with and without basements.

# In[26]:


# Create column for new variable 'HasBsmt'
# if area>0 it gets 1, for area==0 it gets 0
train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
train['HasBsmt'] = 0 
train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1


# In[27]:


# Transform data
train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

# Histogram and normal probability plot
sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


# ### Homoscedasticity

# The best approach to test homoscedasticity for two metric variables is graphically. Departures from an equal dispersion are shown by such shapes as cones (small dispersion at one side of the graph, large dispersion at the opposite side) or diamonds (a large number of points at the center of the distribution).

# #### SalePrice and GrLivArea

# In[28]:


# Scatter plot
plt.scatter(train['GrLivArea'], train['SalePrice']);


# #### SalePrice and TotalBsmt

# In[29]:


# Scatter plot
plt.scatter(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], train[train['TotalBsmtSF']>0]['SalePrice']);


# We note that SalePrice' exhibit equal levels of variance across the range of 'TotalBsmtSF' and 'GrLivArea'. Yay!

# In[30]:


# Convert categorical variable into dummy
train = pd.get_dummies(train)


# ## Models

# #### Data Preparation

# In[31]:


# Split the train set into a new train set and a cross-validation set
split_size = int(train.shape[0]*0.70)
train_x = train[:split_size]
val_x = train[split_size:]
train_y, val_y = train.SalePrice.values[:split_size], train.SalePrice.values[split_size:]

# Prepare the train and test set
X_train = train_x.drop("SalePrice", axis=1)
Y_train = train_x["SalePrice"]
X_test  = val_x.drop("SalePrice", axis=1).copy()
Y_test  = val_y
X_train.shape, Y_train.shape, X_test.shape


# ### Decision Tree

# In[32]:


# Fit the model
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)
acc_decision_tree


# In[33]:


# Add parameters
decision_tree = DecisionTreeRegressor(splitter='best', max_depth=9)
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)
acc_decision_tree


# #### Feature Importance

# In[34]:


featimp = pd.Series(decision_tree.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print (featimp)


# ### XGBoost

# In[35]:


# Import XGBoost model
from xgboost import XGBRegressor

# Fit the model
xgb = XGBRegressor(max_depth=12, reg_lambda=0.5)
xgb.fit(X_train, Y_train)
Y_pred = xgb.predict(X_test)
acc_xgb = round(xgb.score(X_test, Y_test) * 100, 2)
acc_xgb


# In[36]:


# Adjust the parameters 
xgb = XGBRegressor(max_depth=15, reg_lambda=0.4, n_estimators=120)
xgb.fit(X_train, Y_train)
Y_pred = xgb.predict(X_test)
acc_xgb = round(xgb.score(X_test, Y_test) * 100, 2)
acc_xgb


# In[37]:


featimp = pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print (featimp)

