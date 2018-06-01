import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

df = pd.read_csv('final4.csv')


# drop date , unnamed axis ( came in as index when i outputted the file )

df = df.drop('Date',axis=1)
df = df.drop('Unnamed: 0',axis=1)
#remove the first 7500 rows . 
df = df[7500:]

df = df.fillna(0)
image_out_num = len(list(df)) -1 

def normalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    for i in range(0,len(list(df))):
        df[list(df)[i]] = min_max_scaler.fit_transform(df[list(df)[i]].values.reshape(-1,1))
    
    return df



def load_data(stk, seq_len):
    data_raw = stk.as_matrix() # convert to numpy array
    print('created as matrix')
    data = []
    
    # create all possible sequences of length seq_len
    
    for index in range(len(data_raw) - seq_len): 
        data.append(data_raw[index: index + seq_len])
    print('appended all sequence lengths')
    
    data = np.array(data)
    valid_set_size = int(np.round(10/100*data.shape[0]));  
    test_set_size = int(np.round(10/100*data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
    y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]
    
    x_test = data[train_set_size+valid_set_size:,:-1,:]
    y_test = data[train_set_size+valid_set_size:,-1,:]
    
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]




df_norm = normalize_data(df.copy())
print('data normalization done')
seq_len = 7 # choose sequence length
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df_norm, seq_len)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ',x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ',y_test.shape)



index_in_epoch = 0;
perm_array  = np.arange(x_train.shape[0])
np.random.shuffle(perm_array)

# function to get the next batch
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array   
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array) # shuffle permutation array
        start = 0 # start next epoch
        index_in_epoch = batch_size
        
    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]

# parameters
n_steps = seq_len-1 
n_inputs = image_out_num + 1 
n_neurons = 256
n_outputs = image_out_num + 1 
n_layers = 2
learning_rate = 0.01
batch_size = 64
n_epochs = 175
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]
rmse_train_list = []
rmse_test_list = []
rmse_valid_list = []

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

# use Basic RNN Cell
# layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
#           for layer in range(n_layers)]

# use Basic LSTM Cell 
# layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
#          for layer in range(n_layers)]

# use LSTM Cell with peephole connections
# layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons, activation=tf.nn.leaky_relu, use_peepholes = True)
#          for layer in range(n_layers)]

# use GRU cell
layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
         for layer in range(n_layers)]

# layers1 = tf.contrib.rnn.LSTMCell(num_units=n_neurons,activation=tf.nn.leaky_relu, use_peepholes = True)

# layers2 = tf.contrib.rnn.LSTMCell(num_units=n_neurons,activation=tf.nn.leaky_relu, use_peepholes = True)

# layers3 = tf.contrib.rnn.LSTMCell(num_units=n_neurons,activation=tf.nn.leaky_relu, use_peepholes = True)

multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons]) 
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
outputs = outputs[:,n_steps-1,:] # keep only last output of sequence
                                              
loss = tf.reduce_mean(tf.square(outputs - y)) 
loss2 = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(outputs,y),y))) * 100 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
training_op = optimizer.minimize(loss)
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100                                          
# run graph
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess: 
    sess.run(tf.global_variables_initializer())
    for iteration in range(int(n_epochs*train_set_size/batch_size)):
        x_batch, y_batch = get_next_batch(batch_size) # fetch the next training batch 
        sess.run(training_op, feed_dict={X: x_batch, y: y_batch}) 
        if iteration % int(5*train_set_size/batch_size) == 0:
            rmse_train = math.sqrt(loss.eval(feed_dict={X: x_train, y: y_train})) 
            rmse_valid = math.sqrt(loss.eval(feed_dict={X: x_valid, y: y_valid}) )
            rmse_test = math.sqrt(loss.eval(feed_dict={X: x_test, y: y_test}))
            rmse_train_list.append(rmse_train)
            rmse_test_list.append(rmse_test)
            rmse_valid_list.append(rmse_valid)
            mape_train = (loss2.eval(feed_dict={X: x_train, y: y_train}))
            

            print('%.2f epochs: RMSE train/test/valid = %.6f/%.6f/%.6f/%.6f'%(
                int(iteration*batch_size/train_set_size), rmse_train, rmse_test,rmse_valid,mape_train))

    y_train_pred = sess.run(outputs, feed_dict={X: x_train})
    y_valid_pred = sess.run(outputs, feed_dict={X: x_valid})
    y_test_pred = sess.run(outputs, feed_dict={X: x_test})
    
# GridsearchCV not giving any outputs at all , figure out why . 
# implementing denoising autoencoders https://blog.keras.io/building-autoencoders-in-keras.html
# ^ sample code : https://github.com/tgjeon/Keras-Tutorials/blob/master/06_autoencoder.ipynb 


ft = image_out_num
plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test.shape[0]),
         y_test[:,ft], color='black', label='test target',linewidth=2)

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction',linewidth=2)
plt.title('test values chart ')
plt.xlabel('date')
plt.ylabel('log ret')
plt.legend(loc='best')
# plt.rcParams["figure.figsize"] = [20,20]
# plt.show()
plt.savefig("/scratch/sdonthin/4_12.pdf")
plt.close()

def func1():

    plt.plot(rmse_train_list,label = 'train')

    plt.plot(rmse_valid_list, label = 'valid')

    plt.plot(rmse_test_list,label = 'test')
    plt.legend(loc='best')
    plt.gca().invert_xaxis()
    plt.savefig("/scratch/sdonthin/4_11.pdf")
func1()

print('RMSE for training:',     rmse_train)
print('RMSE for testing:',      rmse_test)
print("RMSE for validation : ", rmse_valid)

