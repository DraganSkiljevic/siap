import os
import csv
import math
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score

data_path = "data1.csv"
data_raw = pd.read_csv(data_path)

train, test = train_test_split(data_raw, random_state=42, test_size=0.20, shuffle=True)

train = train.to_numpy()
test = np.array(test)

train_y = train[:, 53]
test_y = test[:, 53]

train_x = np.delete(train, 53, 1)
test_x = np.delete(test, 53, 1)

train_x = train_x.astype('float32')
test_x = test_x.astype('float32')

train_x[np.isnan(train_x)] = 0
test_x[np.isnan(test_x)] = 0

train_x = train_x.astype('int32')
test_x = test_x.astype('int32')

y_label_train = list()

for i in train_y:
    p = str(i).split()
    y_label_train.append(tuple(dict.fromkeys(p)))

y_label_test = list()

for i in test_y:
    p = str(i).split()
    y_label_test.append(tuple(dict.fromkeys(p)))

mlb = MultiLabelBinarizer()

train_y = mlb.fit_transform(y_label_train)
print("klase")
print(mlb.classes_)
print(train_y)

test_y = mlb.fit_transform(y_label_test)

classifier_new = MLkNN(k=10)
# Note that this classifier can throw up errors when handling sparse matrices.
x_train = lil_matrix(train_x).toarray()
y_train = lil_matrix(train_y).toarray()
x_test = lil_matrix(test_x).toarray()

filename = 'model.sav'
start = time.time()

# train
# classifier_new.fit(x_train, y_train)

# save
# pickle.dump(classifier_new, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
print('training time taken: ', round(time.time() - start, 0), 'seconds')
# predict
predictions_new = loaded_model.predict(x_test)
# accuracy
print("Accuracy = ", accuracy_score(test_y, predictions_new))
print("\n")
print("Accuracy = ", hamming_loss(test_y, predictions_new))
print("Accuracy = ", jaccard_score(test_y, predictions_new, average='micro'))
print("Accuracy = ", jaccard_score(test_y, predictions_new, average='macro'))
print("Accuracy = ", jaccard_score(test_y, predictions_new, average='weighted'))
print("Accuracy = ", jaccard_score(test_y, predictions_new, average='samples'))


# print(predictions_new)
# print(test_y)
