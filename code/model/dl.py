
# coding: utf-8

# In[10]:


import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm
import numpy as np
import pandas as pd


# In[11]:


data = pd.read_csv('dataset1.csv')
data = data[7500:]
X = data['log_ret'].values


# In[12]:


size = int(len(X) * 0.8)
train, test = X[0:size], X[int(len(X) * 0.9):len(X)]


# In[23]:


listrep = [x for x in train]
predictions = list()

for t in tqdm(range(1,len(test))):
    model = ARIMA(endog = listrep, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yout = output[0]
    predictions.append(yout)
    obs = test[t]
    listrep.append(obs)
    #print('predicted=%f, expected=%f' % (yhat, obs))
    error = sqrt(mean_squared_error(test[0:t], predictions))
    print('Test RMSE: %.8f' % error)


# # In[ ]:


# pyplot.plot(test)
# pyplot.plot(predictions, color='red')
# pyplot.show()


# # In[22]:





# # In[71]:


# model_fit.summary()

