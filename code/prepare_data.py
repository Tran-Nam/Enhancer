import numpy as np 
import pickle 

data_path = '../data/'
def load_data(path):
    text = open(data_path + path, 'r').readlines()[1::2]
    text = list(map(lambda x: x.strip().upper(), text))
    return text

non_enh_path = 'data_non_enhancers.txt'
strong_enh_path = 'data_strong_enhancers.txt'
weak_enh_path = 'data_week_enhancers.txt'

# load data
non_enh = load_data(non_enh_path)
strong_enh = load_data(strong_enh_path)
weak_enh = load_data(weak_enh_path)
enh = strong_enh + weak_enh
data = enh + non_enh

# vectorize sequence ADN
chars = ['A', 'C', 'G', 'T']
def vectorize(sequence):
    vector = []
    for letter in sequence:
        vector += [0 if letter!=char else 1 for char in chars]
    return vector

# X data
X = []
for sequence in data:
    X.append(vectorize(sequence))
X = np.array(X)

# y data
y = np.zeros((len(data), 1))
y[range(len(enh))] = 1

#print(y[1480:1490])

# print(X.shape)
# print(y.shape)

# split to k folds
def split_ids(num_folds, X_train):
    row_ids = np.random.permutation(range(X_train.shape[0]))
    
    # np.split require equal division
    residual_part = len(row_ids)%num_folds
    val_ids = np.split(row_ids[:len(row_ids) - residual_part], num_folds)
    val_ids[-1] = np.append(val_ids[-1], row_ids[len(row_ids) - residual_part:])
    train_ids = [[k for k in row_ids if k not in val_ids[i]] for i in range(num_folds)]
    
    data = []
    for i in range(num_folds):
        data.append([train_ids[i], val_ids[i]])
    return data
split_ids = split_ids(num_folds=5, X_train=X)
# for i in range(len(data)):
#     print('#######')
#     print(len(data[i][0]))
#     print(len(data[i][1]))

data = {'X': X, 'y': y, 'ids': split_ids}

with open(data_path + 'prepare_data.pkl', 'wb') as f:
    pickle.dump(data, f)