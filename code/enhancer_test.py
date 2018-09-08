import numpy as np 
import pickle
#from enhancer_detect import predict

with open('../data/test_data.pkl', 'rb') as f:
    data_test = pickle.load(f)

X_test = data_test['X']
y_test = data_test['y']
y_test = np.array(y_test).reshape(len(y_test), 1)


data_path = '../data/prepare_data.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

X_train = data['X']
y_train = data['y']
ids = data['ids']


#print(X_test[0])
#print(y_test[0])
#print(y_test.shape)
#print(X_test.shape)
#print(y_test.shape)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict(w1, b1, w2, b2, X):
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    return a2  #np.max(a2, axis=1, keepdims=True)



num_model = 5
y_pred_mean = np.zeros_like(y_test)
y_pred_vote = np.zeros_like(y_test)
for i in range(num_model):
    #print('Model : ' + str(i+1))
    #y_pred = np.zeros_like(y_test)
    with open('./model/model_{0}.pkl'.format(i+1), 'rb') as f:
        model = pickle.load(f)
    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
    pred_mean = predict(w1, b1, w2, b2, X_test)
    pred_vote = (pred_mean >= 0.5).astype(int) #vote
    
    
    val_part = {'X': X_train[ids[i][1]], 'y': y_train[ids[i][1]]}
    y_pred = predict(w1, b1, w2, b2, val_part['X'])
    pred = (y_pred >= 0.5).astype(int)
    #print('Acc_val: %.2f'%(100*(np.mean(pred_ == val_part['y']))))
    #print(pred.shape)
    #print(pred[5:10])
    y_pred_mean += pred_mean
    y_pred_vote += pred_vote
    #print(y_pred[390:400])

y_pred_mean /= num_model
y_pred_vote /= num_model
#print(y_pred.shape)
y_pred_mean = (y_pred_mean >= 0.5).astype(int)
y_pred_vote = (y_pred_vote >= 0.5).astype(int)
#print(y_pred[350:400])
#print(y_test[350:400])

print('Acc by mean: %.2f'%(100*np.mean(y_pred_mean == y_test.astype(int))) + '%')
print('Acc by vote: %.2f'%(100*np.mean(y_pred_vote == y_test.astype(int))) + '%')