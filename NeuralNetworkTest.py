import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



#Reading the training and testing data
X_train = np.loadtxt('X_train.txt')
Y_train = np.loadtxt('y_train.txt')
X_test = np.loadtxt('X_test.txt')
Y_test = np.loadtxt('y_test.txt')
print('x_train shape:',X_train.shape)
print('y_train shape:',Y_train.shape)
print('x_test shape:',X_test.shape)
print('y_test shape:',Y_test.shape)

#Normalizing the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
print(X_train)
print(X_train.shape)
X_test = scaler.transform(X_test)
print(X_test.shape)

#PCA
# pca = PCA(n_components=500)
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# print(X_train.shape)
# X_test = pca.transform(X_test)
# print(X_test.shape)


#HL and alpha to be changed based on the results from the past two files
clf = MLPClassifier(hidden_layer_sizes=(700,),alpha=1)

clf.fit(X_train,Y_train)

predictions = clf.predict(X_test)

print(classification_report(Y_test, predictions))