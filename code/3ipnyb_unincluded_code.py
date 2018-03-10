# corr_price_development_train = np.sum(np.equal(np.sign(y_train[:,1]-y_train[:,0]),
#             np.sign(y_train_pred[:,1]-y_train_pred[:,0])).astype(int)) / y_train.shape[0]
# corr_price_development_valid = np.sum(np.equal(np.sign(y_valid[:,1]-y_valid[:,0]),
#             np.sign(y_valid_pred[:,1]-y_valid_pred[:,0])).astype(int)) / y_valid.shape[0]
# corr_price_development_test = np.sum(np.equal(np.sign(y_test[:,1]-y_test[:,0]),
#             np.sign(y_test_pred[:,1]-y_test_pred[:,0])).astype(int)) / y_test.shape[0]

# print('correct sign prediction train/valid/test: %.2f/%.2f/%.2f'%(
#     corr_price_development_train, corr_price_development_valid, corr_price_development_test))




# ## show predictions
# plt.figure(figsize=(15, 5));
# plt.subplot(1,2,1);

# plt.plot(np.arange(y_train.shape[0]), y_train[:,ft], color='blue', label='train target')

# plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_valid.shape[0]), y_valid[:,ft],
#          color='gray', label='valid target')

# plt.plot(np.arange(y_train.shape[0]+y_valid.shape[0],
#                    y_train.shape[0]+y_test.shape[0]+y_test.shape[0]),
#          y_test[:,ft], color='black', label='test target')

# plt.plot(np.arange(y_train_pred.shape[0]),y_train_pred[:,ft], color='red',
#          label='train prediction')

# plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_valid_pred.shape[0]),
#          y_valid_pred[:,ft], color='orange', label='valid prediction')

# plt.plot(np.arange(y_train_pred.shape[0]+y_valid_pred.shape[0],
#                    y_train_pred.shape[0]+y_valid_pred.shape[0]+y_test_pred.shape[0]),
#          y_test_pred[:,ft], color='green', label='test prediction')

# plt.title('train test validate')
# plt.xlabel('date')
# plt.ylabel('log ret')
# plt.legend(loc='best');

# plt.subplot(1,2,2);



