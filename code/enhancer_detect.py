import numpy as np 
import matplotlib.pyplot as plt 
import pickle 

data_path = '../data/prepare_data.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

X_train = data['X']
y_train = data['y']
ids = data['ids']

#for i in range(len(ids)):
#    print('Fold: ' + str(i+1))
#    print(y_train[ids[i][1]].shape)
#    print(y_train[ids[i][1]][:10])
# for i in range(len(ids)):
#     print('########')
#     print(len(ids[i][0]))
#     print(len(ids[i][1]))

# visualize data
# fig = plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(X_train[ids[0][0][i]].reshape(25, 32).T, cmap=plt.cm.binary_r)
#     plt.show()
    

# activation
def sigmoid(z):
    return 1/(1+np.exp(-z))

# gradient
def grad_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

# cost
def cost(y_pred, y_target):
    num_train = y_target.shape[0]
    return -np.sum(y_target*np.log(y_pred) + (1-y_target)*np.log(1-y_pred)) / num_train

# add regularization
def cost_reg(y_pred, y_target, w1, w2, reg):
    num_train = y_target.shape[0]
    return -np.sum(y_target*np.log(y_pred) + (1-y_target)*np.log(1-y_pred)) / num_train + \
            reg*(np.sum(w1*w1)+np.sum(w2*w2)) / (2*num_train)

# hyperparameter
num_hiddens = [32, 64, 128]
lrs = [.1, .3, 1]
regs = [.1, .5, 1, 4, 16, 32, 64, 128]

# num_hiddens = [128]
# lrs = [.1]
# regs = [32]

# train 
def train(X_train, y_train, X_val, y_val, num_hidden, lr, reg):
    
    num_train = X_train.shape[0]
    num_feature = X_train.shape[1]

    # initial parameter
    w1 = np.random.rand(num_feature, num_hidden) * np.sqrt(2/(num_feature+num_hidden))
    b1 = np.zeros((1, num_hidden))
    w2 = np.random.rand(num_hidden, 1) * np.sqrt(2/num_hidden)
    b2 = np.zeros((1, 1))
    loss_value = []
    loss_val_value = []
    # e = 1e-8

    for it in range(ite):
        # forward
        z1 = np.dot(X_train, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = sigmoid(z2)

        #compute losssssssssssssssssssssssssssssss
        loss = cost(a2, y_train)
        pred = predict(w1, b1, w2, b2, X_val)
        loss_val = cost(pred, y_val)
        if it%100 == 0:
            print('Ite {0} Loss {1:.4f}'.format(it, loss))
        if it%10 == 0:
            loss_value.append(loss)
            loss_val_value.append(loss_val)
        

        

        #backward
        dz2 = (a2 - y_train) / num_train
        dw2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = np.dot(dz2, w2.T)
        dz1 = da1*grad_sigmoid(z1)
        dw1 = np.dot(X_train.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        #regu
        dw1 += reg*w1/num_train
        dw2 += reg*w2/num_train

        #if np.linalg.norm(dw1) < e and np.linalg.norm(dw2) < e and \
        #    np.linalg.norm(db1) < e and np.linalg.norm(db1) < e:
        #    break

        #update
        w1 -= lr*dw1
        b1 -= lr*db1
        w2 -= lr*dw2
        b2 -= lr*db2
    return w1, b1, w2, b2, loss_value, loss_val_value

# predict
def predict(w1, b1, w2, b2, X):
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    return a2

ite = 1200
# cross validation to choose parameter
def choose_param(best_param, min_loss, num_hiddens, lrs, regs):
    for num_hidden in num_hiddens:
        for lr in lrs:
            for reg in regs:
                param = [num_hidden, lr, reg]
                print('###############')
                print('{0} hidden, {1:.2f} lr, {2:.2f} reg'.format(num_hidden, lr, reg))
                aver_loss = 0
                # loss_train = []
                loss_train = []
                loss_val = []
                for i in range(len(ids)):
                    print('Fold : {0}'.format(i))
                    train_part = {'X': X_train[ids[i][0]], 'y': y_train[ids[i][0]]}
                    val_part = {'X': X_train[ids[i][1]], 'y': y_train[ids[i][1]]}
                    # print(train_part['X'].shape, train_part['y'].shape)
                    #train model
                    w1, b1, w2, b2, loss_train_value, loss_val_value = train(train_part['X'], train_part['y'], val_part['X'], val_part['y'], num_hidden, lr, reg)

                    #visual
                    # loss_train += loss_train_value
                    # loss_val += loss_val_value
                    xplot = np.arange(0, ite, 10)
                    #fig = plt.figure(figsize=(10, 10))
                    # plt.subplot(len(ids), len(ids), i+1)
                    #plt.plot(xplot, loss_train_value, label='Loss_train')
                    #plt.plot(xplot, loss_val_value, label='loss_val')
                    #plt.legend()
                    #plt.savefig('./fig/{0}_{1:.2f}_{2:.2f}_fold-{3}.png'.format(num_hidden, lr, reg, i+1))

                    y_pred = predict(w1, b1, w2, b2, val_part['X'])
                    pred = (y_pred >= 0.5).astype(int)
                    # print(pred[:5])
                    # print(val_part['y'][:5])
                    print('Acc_val: %.2f'%(100*(np.mean(pred == val_part['y']))))
                    aver_loss += cost(y_pred, val_part['y'])
                # loss_train = [i/len(ids) for i in loss_train]
                # loss_val = [i/len(ids) for i in loss_val]

                # xplot = np.arange(0, ite, 10)
                # fig = plt.figure(figsize=(10, 10))
                # plt.plot(xplot, loss_train, label='Loss_train')
                # plt.plot(xplot, loss_val, label='Loss_val')
                # plt.legend()
                # plt.savefig('./fig/{0}_{1:.2f}_{2:.2f}.png'.format(num_hidden, lr, reg))
                # aver_loss /= len(ids)
                # loss_train /= len(ids)
                # xplot = np.arange(0, ite, 10)
                # fig = plt.figure(figsize=(10, 10))
                # plt.plot(xplot, loss_train)
                # plt.plot(xplot, aver_loss)
                # plt.savefig('./fig/{0}_{1}_{2}.png'.format(num_hidden, lr, reg))
                
                aver_loss /= len(ids)
                print('Aver_loss: {0: .4f}'.format(aver_loss))
                
                param = [num_hidden, lr, reg]
                

                #compare loss
                if aver_loss < min_loss:
                    min_loss = aver_loss
                    best_param = param

    return best_param

# choose param
best_param = choose_param([1, 1, 1], 10, num_hiddens, lrs, regs)
#print('Best param: ', best_param)

# build model (ensemble, 5 model)
num_hidden, lr, reg = best_param
param = {'num_unit_hidden': num_hidden, 'learning_rate': lr, 'regularization': reg}
with open('./model/param.pkl', 'wb') as f:
    pickle.dump(param, f)

print('########') 
print('########')
print('########')
print('Best param : {0} hidden, {1: .3f} lr, {2: .3f} reg'.format(num_hidden, lr, reg))


for i in range(len(ids)):
    print('Fold : {0}'.format(i))
    train_part = {'X': X_train[ids[i][0]], 'y': y_train[ids[i][0]]}
    val_part = {'X': X_train[ids[i][1]], 'y': y_train[ids[i][1]]}
        # train model
    w1, b1, w2, b2, _, __ = train(train_part['X'], train_part['y'],val_part['X'], val_part['y'], num_hidden, lr, reg)
    model = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    with open('./model/model__{0}.pkl'.format(i+1), 'wb') as f:
        pickle.dump(model, f)