import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
# pca = PCA(n_components=5)
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# print(X_train.shape)

def mlpPlot():
    param_range_C = [0.1, 1, 10, 100, 1000]
    #change gamma based on the past answer
    train_scores, valid_scores = validation_curve(SVC(random_state=101, gamma=0.001), X_train, Y_train, 'C', param_range_C, cv = 3, verbose = True, n_jobs = -1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)
    plt.title("Validation Curve with SVM PCA(5)")
    plt.xlabel("C")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range_C, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range_C, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range_C, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range_C, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

if __name__ == '__main__':
    mlpPlot()