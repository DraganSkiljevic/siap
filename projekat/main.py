import os
import csv
import math
import pandas as pd
import numpy as np
import pickle
import time
from matplotlib import pyplot as plt
from scipy.linalg.tests.test_fblas import accuracy
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score, f1_score
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain


def accuracy(test, predictions_new):
    # accuracy

    print("Accuracy score = ", accuracy_score(test, predictions_new))
    print('Accuracy F1-score:', f1_score(test, predictions_new, average='micro'))
    print("Accuracy hamming loss= ", hamming_loss(test, predictions_new))
    print("Accuracy jaccard micro= ", jaccard_score(test, predictions_new, average='micro'))
    print("Accuracy jaccard macro= ", jaccard_score(test, predictions_new, average='macro'))
    print("Accuracy jaccard weighted= ", jaccard_score(test, predictions_new, average='weighted'))
    print("Accuracy jaccard samples= ", jaccard_score(test, predictions_new, average='samples'))


def multinomialLogisticRegression():
    # Train multi-classification model with logistic regression
    print("Logisticka regresija")

    start = time.time()
    classifier = BinaryRelevance(
        classifier=linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg'),
        require_dense=[False, True]
    )

    filename = "logistickaRegresija.sav"
    classifier.fit(train_x, train_y)

    # save
    pickle.dump(classifier, open(filename, 'wb'))

    # load the model from disk
    classifier = pickle.load(open(filename, 'rb'))

    print('training time taken: ', round(time.time() - start, 0), 'seconds')

    predictions_new = classifier.predict(test_x)

    accuracy(test_y, predictions_new)


def multinomialLogisticRegressionChain():
    # Train multi-classification model with logistic regression
    print("Logisticka regresija chain")

    start = time.time()
    classifier = BinaryRelevance(
        classifier=linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg'),
        require_dense=[False, True]
    )

    filename = "logistickaRegresija.sav"
    classifier.fit(train_x, train_y)

    # save
    pickle.dump(classifier, open(filename, 'wb'))

    # load the model from disk
    classifier = pickle.load(open(filename, 'rb'))

    print('training time taken: ', round(time.time() - start, 0), 'seconds')

    predictions_new = classifier.predict(test_x)

    accuracy(test_y, predictions_new)


def randomForest():
    print("Random forest classifier")

    start = time.time()
    classifier = BinaryRelevance(
        classifier=RandomForestClassifier(),
        require_dense=[False, True]
    )
    filename = "randomForest"

    classifier.fit(train_x, train_y)

    # save
    pickle.dump(classifier, open(filename, 'wb'))

    # load the model from disk
    classifier = pickle.load(open(filename, 'rb'))

    print('training time taken: ', round(time.time() - start, 0), 'seconds')

    predictions_new = classifier.predict(test_x)

    accuracy(test_y, predictions_new)


def randomForestClassifierChain():
    print("Random forest classifier chain")

    start = time.time()
    classifier = ClassifierChain(
        classifier=RandomForestClassifier(),
        require_dense=[False, True]
    )
    filename = "randomForestClassifierChain"

    classifier.fit(train_x, train_y)

    # save
    pickle.dump(classifier, open(filename, 'wb'))

    # load the model from disk
    classifier = pickle.load(open(filename, 'rb'))

    print('training time taken: ', round(time.time() - start, 0), 'seconds')

    predictions_new = classifier.predict(test_x)

    accuracy(test_y, predictions_new)


def gaussianNaiveBayesBinary():
    print("Gaussian naive bayes binary")

    start = time.time()
    classifier = BinaryRelevance(GaussianNB())

    filename = "gaussianNaiveBayes"

    classifier.fit(train_x, train_y)

    # save
    pickle.dump(classifier, open(filename, 'wb'))

    # load the model from disk
    classifier = pickle.load(open(filename, 'rb'))

    print('training time taken: ', round(time.time() - start, 0), 'seconds')

    predictions_new = classifier.predict(test_x)

    accuracy(test_y, predictions_new)


def gaussianNaiveBayes():
    print("Gaussian naive bayes")

    start = time.time()
    classifier = ClassifierChain(GaussianNB())

    filename = "gaussianNaiveBayes"

    classifier.fit(train_x, train_y)

    # save
    pickle.dump(classifier, open(filename, 'wb'))

    # load the model from disk
    classifier = pickle.load(open(filename, 'rb'))

    print('training time taken: ', round(time.time() - start, 0), 'seconds')

    predictions_new = classifier.predict(test_x)

    accuracy(test_y, predictions_new)


def knnClassifierChain():
    print("knn classifier chain")

    start = time.time()
    classifier = ClassifierChain(KNeighborsClassifier())

    filename = "knnChain"

    classifier.fit(train_x, train_y)

    # save
    pickle.dump(classifier, open(filename, 'wb'))

    # load the model from disk
    classifier = pickle.load(open(filename, 'rb'))

    print('training time taken: ', round(time.time() - start, 0), 'seconds')

    predictions_new = classifier.predict(test_x)

    accuracy(test_y, predictions_new)


def knnBinary(m):
    print("knn binary")

    start = time.time()
    classifier = BinaryRelevance(KNeighborsClassifier(n_neighbors=m))

    filename = "knnBinary"

    classifier.fit(train_x, train_y)

    # save
    pickle.dump(classifier, open(filename, 'wb'))

    # load the model from disk
    classifier = pickle.load(open(filename, 'rb'))

    print('training time taken: ', round(time.time() - start, 0), 'seconds')

    predictions_new = classifier.predict(test_x)

    accuracy(test_y, predictions_new)


def supportVectorMachine():
    print("Support vector machine")

    start = time.time()
    classifier = BinaryRelevance(
        classifier=svm.SVC(),
        require_dense=[False, True]
    )
    filename = "SupportVectorMachine"

    classifier.fit(train_x, train_y)

    # save
    pickle.dump(classifier, open(filename, 'wb'))

    # load the model from disk
    classifier = pickle.load(open(filename, 'rb'))

    print('training time taken: ', round(time.time() - start, 0), 'seconds')

    predictions_new = classifier.predict(test_x)

    accuracy(test_y, predictions_new)


def supportVectorMachineChain():
    print("Support vector machine")

    start = time.time()
    classifier = ClassifierChain(
        classifier=svm.SVC(),
        require_dense=[False, True]
    )
    filename = "SupportVectorMachine"

    classifier.fit(train_x, train_y)

    # save
    pickle.dump(classifier, open(filename, 'wb'))

    # load the model from disk
    classifier = pickle.load(open(filename, 'rb'))

    print('training time taken: ', round(time.time() - start, 0), 'seconds')

    predictions_new = classifier.predict(test_x)

    accuracy(test_y, predictions_new)


def multiLabelKnn():
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


def randomForestClassifierChain():
    print("Random forest classifier chain")

    start = time.time()
    classifier = ClassifierChain(
        classifier=RandomForestClassifier(),
        require_dense=[False, True]
    )
    filename = "randomForestClassifierChain"

    # classifier.fit(train_x, train_y)

    # save
    # pickle.dump(classifier, open(filename, 'wb'))

    # load the model from disk
    classifier = pickle.load(open(filename, 'rb'))

    print('training time taken: ', round(time.time() - start, 0), 'seconds')

    predictions_new = classifier.predict(test_x)

    accuracy(test_y, predictions_new)


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

train_x = train_x / train_x.max(axis=0)
test_x = test_x / test_x.max(axis=0)

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

randomForestClassifierChain()
# randomForest()
# gaussianNaiveBayes()

test_y = mlb.fit_transform(y_label_test)

# print(predictions_new)
# print(test_y)

multiLabelKnn()
randomForest()
randomForestClassifierChain()
supportVectorMachine()
supportVectorMachineChain()
multinomialLogisticRegression()
multinomialLogisticRegressionChain()
gaussianNaiveBayes()
gaussianNaiveBayesBinary()
knnClassifierChain()
knnBinary(3)
# knnBinary(21)
# knnBinary(4)
# knnBinary(5)


# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(train_x)
#     print(i)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()
#
# f = plt.figure(figsize=(19, 15))
# plt.matshow(data_raw.corr(), fignum=f.number)
# plt.xticks(range(data_raw.shape[1]), data_raw.columns, fontsize=14, rotation=90)
# plt.yticks(range(data_raw.shape[1]), data_raw.columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16);
# plt.show()

# x = [2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 19, 21]
# y = [74.58, 79.19, 78.79, 80.67, 80.35, 81.40, 81,24, 81.87, 82.24, 82.61, 82.53,82.68,]

# plotting the points
# plt.plot(x, y)
#
# # naming the x axis
# plt.xlabel('k')
# # naming the y axis
# plt.ylabel('preciznost')
#
# # giving a title to my graph
# plt.title('Knn binary relevance')
#
# # function to show the plot
# plt.show()

# plt.scatter(kick, position)
# plt.ylabel('GK reflexes', fontsize=18)
# plt.xlabel('GK positioning', fontsize=18)
# plt.show()
#
# datapom = np.array(datapom)
# p = data_raw.shape
# jedan = 0
# dva = 0
# tri = 0
# cetiri = 0
# pet = 0

# for i in range(p[0]):
#     pp = str(datapom[i, 46]).split()
#     pp = len(pp)
#     if pp == 1:
#         jedan += 1
#     if pp == 2:
#         dva += 1
#     if pp == 3:
#         tri += 1
#     if pp == 4:
#         cetiri += 1
#     if pp == 5:
#         pet += 1
#
# print(jedan)
# print(dva)
# print(tri)
# print(cetiri)
# print(pet)

# sn.set(font_scale=2)
# plt.figure(figsize=(15, 8))
# ax = sn.barplot(mlb.classes_, [3630, 8521, 2029, 3397, 4957, 3763])
# plt.title("Broj igraca po pozicijama", fontsize=24)
# plt.ylabel('Broj torki', fontsize=18)
# plt.xlabel('Pozicije', fontsize=18)
# # adding the text labels
# rects = ax.patches
# labels = [3630, 8521, 2029, 3397, 4957, 3763]
# for rect, label in zip(rects, labels):
#     height = rect.get_height()
#     ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom', fontsize=18)
# plt.show()

# sn.set(font_scale=2)
# plt.figure(figsize=(15, 8))
# ax = sn.barplot(["jedan", "dva", "tri", "cetiri"], [9738, 5897, 2001, 344])
# plt.title("Broj igraca koji igraju na vise pozicija", fontsize=24)
# plt.xlabel('Broj pozicija', fontsize=18)
# # adding the text labels
# rects = ax.patches
# labels = [9738, 5897, 2001, 344]
# for rect, label in zip(rects, labels):
#     height = rect.get_height()
#     ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom', fontsize=18)
# plt.show()