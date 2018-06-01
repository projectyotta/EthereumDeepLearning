import matplotlib
matplotlib.use('Agg')
from math import sqrt
from numpy import concatenate
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import LSTM,GRU
from keras import backend as K
from keras.layers import TimeDistributed
from keras import optimizers
    




# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
import math 

# load dataset
dataset = read_csv('truncatedkeras.csv', header=0, index_col=0)

meanval = np.mean(dataset['log_ret'])
for i in range(0,len(dataset)):
    if(dataset['log_ret'][i]>meanval):
        dataset['log_ret'][i] == meanval 
# try adding the least log ret value so that least ossible value is zero ? 
#dataset['log_ret'] = abs(dataset['log_ret'].min()) + dataset['log_ret']


# x = list(dataset.columns)
# y = ['Date','log_ret']
# y = y+(x)
# print('dataset finished reading')

# df_temp = pd.DataFrame()
# df_temp['cols'] = y 
# df_temp = df_temp[df_temp.index != 16398]
# df_temp = df_temp[df_temp.index != 16386]

# dataset = dataset[df_temp['cols'].tolist()]

dataset = dataset.fillna(0)

# dataset = dataset.round({'log_ret': 4})
# dataset['log_ret'] = dataset['log_ret'] * 0.0001 

# dataset = dataset[7000:]
values = dataset.values




# ensure all data is float
values = values.astype('float32')


# normalize features

# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaled = scaler.fit_transform(values)

scaler = StandardScaler()
scaled = scaler.fit_transform(values)


# scaler = MaxAbsScaler()
# scaled = scaler.fit_transform(values)



# scaler = RobustScaler()
# scaled = scaler.fit_transform(values)






# specify the number of lag hours
n_hours = 5
n_features = len(list(dataset))
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)
print(reframed.shape)

# split into train and test sets
values = reframed.values
n_train_hours = 19200
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# def root_mean_squared_error(y_true, y_pred):
#         return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


# adding hidden layers how to logic 

# model = Sequential()
# model.add(LSTM(..., return_sequences=True, input_shape=(...)))
# model.add(LSTM(..., return_sequences=True))
# model.add(LSTM(..., return_sequences=True))
# model.add(LSTM(...))
# model.add(Dense(...))

# multi layer rnn with lstm 
# In this model I have used 3 layers of LSTM with 512 neurons per layer 
# followed by 0.25 Dropout after each LSTM layer to prevent over-fitting 
# and finally a Dense layer to produce our outputs.
# activ_func = 'tanh'
# neurons = 64
# dropout = 0
# model = Sequential()
# model.add(LSTM(neurons, return_sequences=True,activation=activ_func, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dropout(dropout))
# model.add(LSTM(neurons, return_sequences=True, activation=activ_func))
# model.add(Dropout(dropout))
# model.add(LSTM(neurons, activation=activ_func))
# model.add(Dropout(dropout))
# model.add(Dense(1))
# model.add(Activation(activ_func))
# model.compile(loss='mse', optimizer='adam')
 # , metrics=['mae']


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

import math 

def weighted_rmse1(y_true,y_pred):

    # ones = K.ones_like(y_true[0,:]) #a simple vector with ones shaped as (60,)
    # idx = K.cumsum(ones) #similar to a 'range(1,61)'
    diff = abs((y_pred-y_true)*100)
    ratio = abs(y_pred-y_true)
    return K.sqrt(K.mean((1/(diff*100))*K.square(y_pred-y_true)))


def weighted_rmse2(y_true,y_pred):

    # ones = K.ones_like(y_true[0,:]) #a simple vector with ones shaped as (60,)
    # idx = K.cumsum(ones) #similar to a 'range(1,61)'
    diff = abs((y_pred-y_true)*100)
    ratio = abs(y_true/y_pred) 
    

    #print('printing y pred vals to see how it looks')
    
    #print(y_pred)
    return K.sqrt(K.mean(K.square(ratio*(y_pred - y_true)), axis=-1))
    #return K.sqrt(K.mean(K.square( ratio*(y_pred-y_true))))
    


def weighted_rmse3(y_true,y_pred):

    ratio = y_true
    average = np.mean(test_y)
    for i in range(0,len(test_y)):
    	if(test_y[i]>average):
    		ratio[i] = 10
    	else:
    		ratio[i] = 5 



    #print('printing y pred vals to see how it looks')
    
    #print(y_pred)

    return K.sqrt(K.mean(K.square(ratio * (y_pred-y_true))))

    



# design network
model = Sequential()
model.add(GRU(135, return_sequences=True, activation = 'linear' , input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.5))

# model.add(GRU(135, return_sequences=True))
# model.add(Dropout(0.3))



model.add(GRU(135))
model.add(Dropout(0.5))
#model.add(Dense(1))

model.add((Dense(1, activation='linear'))) # sequence labeling 



learning_rate = 0.01
decay_rate = learning_rate / 100
momentum = 0.8
sgd = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)


rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

# rmsprop = optimizers.RMSprop(lr=0.001)


model.compile(loss=root_mean_squared_error, optimizer=rmsprop)
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=32, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig("/scratch/sdonthin/4_14.pdf")
pyplot.close()


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -(len(list(dataset))-1):]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -(len(list(dataset))-1):]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
# inv_yhat = np.around(inv_yhat, decimals=4)



rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.8f' % rmse)


x = pd.DataFrame()
x['iny_y'] = inv_y
x['inv_yhat'] = inv_yhat

x.to_csv('/scratch/sdonthin/vals.csv',index=False)

pyplot.plot(inv_y, label='invy')
pyplot.plot(inv_yhat, label='invyhat')
pyplot.legend()
pyplot.savefig("/scratch/sdonthin/4_15.pdf")


from keras.models import load_model

model.save('/scratch/sdonthin/my_model1.h5')






