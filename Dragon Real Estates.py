#!/usr/bin/env python
# coding: utf-8

# ## Dragon Real Estate Price Predictor

# In[1]:


import "pandas" as pd 


# In[2]:


hausing = pd.read_csv("data.csv")


# In[3]:


hausing.head()


# In[4]:


hausing.info()


# In[5]:


hausing['CHAS'].value_counts()


# In[6]:


hausing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import matplotlib as plt


# In[9]:


hausing.hist(bins=50, figsize=(20,15))


# ## Train - Test Splitting

# In[10]:


import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[11]:


train_set, test_set = split_train_test(hausing, 0.2)


# In[12]:


print(f"Rows in train set : {len(train_set)}\nRows in test set : {len(test_set)}\n")


# In[13]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(hausing, test_size = 0.2, random_state = 42)
print(f"Rows in train set : {len(train_set)}\nRows in test set : {len(test_set)}\n")


# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(hausing, hausing['CHAS']):
    strat_train_set = hausing.loc[train_index]
    strat_test_set = hausing.loc[test_index]


# In[15]:


strat_test_set['CHAS'].value_counts()


# In[16]:


strat_train_set['CHAS'].value_counts()


# ## Looking for Correlations

# In[17]:


corr_matrix = hausing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[18]:


from pandas.plotting import scatter_matrix
attributes = ['MEDV', 'RM', 'ZN', 'LSTAT']
scatter_matrix(hausing[attributes], figsize = (12, 8))


# In[19]:


hausing.plot(kind='scatter', x='RM', y='MEDV', alpha=0.8)


# ## Trying out Attributr combination

# In[20]:


hausing['TAXRM'] = hausing['TAX']/hausing['RM']


# In[21]:


hausing.head()


# In[22]:


corr_matrix = hausing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[23]:


hausing.plot(kind='scatter', x='TAXRM', y='MEDV', alpha=0.8)


# In[24]:


hausing = strat_train_set.drop('MEDV', axis=1)
hausing_labels = strat_train_set['MEDV'].copy()


# ## Missing Attributes

# In[25]:


median = hausing['RM'].median()


# In[26]:


hausing['RM'].fillna(median)


# In[27]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(hausing)


# In[28]:


X = imputer.transform(hausing)
hausing_tr = pd.DataFrame(X, columns=hausing.columns)
hausing_tr.describe()


# ## Creating a Pipelines

# In[29]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])


# In[30]:


hausing_num_tr = my_pipeline.fit_transform(hausing)


# In[31]:


hausing_num_tr.shape


# ## Selecting a desiered model for Dragon Real Estates

# In[32]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(hausing_num_tr, hausing_labels)


# In[33]:


some_data = hausing.iloc[:5]


# In[34]:


some_labels = hausing.iloc[:5]


# In[35]:


prepared_data = my_pipeline.transform(some_data)


# In[36]:


model.predict(prepared_data)


# In[37]:


some_labels


# In[38]:


from sklearn.metrics import mean_squared_error
hausing_prediction = model.predict(hausing_num_tr)
mse = mean_squared_error(hausing_labels, hausing_prediction)
rmse = np.sqrt(mse)


# In[39]:


mse


# ## Using better evaluation technique - Cross Validation

# In[40]:


from sklearn.model_selection import cross_val_score
scores= cross_val_score(model, hausing_num_tr, hausing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[41]:


rmse_scores


# In[42]:


def print_scores(scores):
    print("Scores : " , scores)
    print("Mean : ", scores.mean())
    print("Standrad deviation : ", scores.std())


# In[43]:


print_scores(rmse_scores)


# ## Saving the Model

# In[44]:


from joblib import dump, load
dump(model, 'Dragon.joblib')


# In[45]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[46]:


final_rmse


# ## Using the Model

# In[52]:


from joblib import dump, load
import numpy as np
model=load('Dragon.joblib')
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24141041, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)


# In[ ]:




