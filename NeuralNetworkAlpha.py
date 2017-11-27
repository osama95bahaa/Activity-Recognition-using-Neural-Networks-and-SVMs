import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
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

#PCA
# pca = PCA(n_components=500)
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# print(X_train.shape)

def mlpPlot():
    param_range_alpha = [0.00001, 0.001, 0.1, 1, 10]
    #hidden layer size to be changed
    train_scores, valid_scores = validation_curve(MLPClassifier(random_state=101, hidden_layer_sizes=(700,)), X_train, Y_train, 'alpha', param_range_alpha, cv = 3, verbose = True, n_jobs = -1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)
    plt.title("Validation Curve with MLP(PCA 500)")
    plt.xlabel("Alpha")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range_alpha, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range_alpha, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range_alpha, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range_alpha, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

if __name__ == '__main__':
    mlpPlot()