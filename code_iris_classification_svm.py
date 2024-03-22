import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split



 #*EXERCICE 1 *

# data loading
iris = datasets.load_iris()

#we keep  the two first attributes
X,y = iris.data[:,:2] , iris.target
# we'll use half of the dataset for evaluation
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5)

# trainig the traing dataset set to a linaer support vector
C=1.0 # C is the regulization parameter
lin_svc = svm.LinearSVC(C=C)
lin_svc.fit(X_train , y_train)

# Computing the score
score1 = lin_svc.score(X_test, y_test)
print("The model's score on the training dataset: ", score1)


# Creation of the discrete decision surface
x_min, x_max = X[:,0].min() -1 , X[:,0].max() +1
y_min , y_max = X[:,1].min()-1 , X[:,1].max()+1


# to display the decision surace we are gonna discretiser l'espace avec un pas h

h = max((x_max - x_min) / 100, (y_max - y_min) / 100)

xx, yy = np.meshgrid(np.arange(x_min, x_max,h),np.arange(y_min, y_max,h))
# decision surface
# Surface de décision 
Z = lin_svc.predict(np.c_[xx.ravel(), yy.ravel()]) 
Z = Z.reshape(xx.shape) 
 
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
# to display the learning points
plt.scatter(X_train[:, 0], X_train[:, 1], label="train", edgecolors='k', 
c=y_train, cmap=plt.cm.coolwarm) 
plt.scatter(X_test[:, 0], X_test[:, 1], label="test", marker='*', c=y_test, cmap=plt.cm.coolwarm)
plt.xlabel('Sepal length') 
plt.ylabel('Sepal width')
plt.title("LinearSVC") 
plt.show()

#svc with linear kernel
lin_svc = svm.LinearSVC(C=C).fit(X_train,y_train)
svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)

fig = plt.figure(figsize=(12, 5))


title = ['SVC with linear kernel', 'LinearSVC (linear kernel)']

for i, clf in enumerate((svc, lin_svc)):
    plt.subplot(1, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Utiliser une palette de couleurs
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # Afficher aussi les points d'apprentissage
    plt.scatter(X_train[:, 0], X_train[:, 1], label="train", edgecolors='k', c=y_train, cmap=plt.cm.coolwarm)
    plt.scatter(X_test[:, 0], X_test[:, 1], label="test", marker='*', c=y_test, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length') 
    plt.ylabel('Sepal width') 
    plt.title(title[i])
plt.show()



#Now we'll train the model for the 4 attributes attributes
X,y = iris.data , iris.target
# we use half of the dataset for evaluation
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5)

# trainig the traing dataset set to a linaer support vector
C=1.0 # C is the regulization parameter
lin_svc = svm.LinearSVC(C=C)
lin_svc.fit(X_train , y_train)

# Computing the score
score2 = lin_svc.score(X_test, y_test)
print("The model's score on the training dataset: ", score2)
















