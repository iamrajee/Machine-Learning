#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:15:34 2018

SUPPORT VECTOR MACHINES
Import required packages

@author: Anish MM
"""
# For math.
import numpy as np
# For plotting.
import matplotlib.pyplot as plt
# for svm.
from sklearn import svm
# Iris dataset (a dataset we will work with).
from sklearn import datasets

#Ignore the following cell
#The following cell is used for plotting. Not related to our topic. Copy this for now. You can work it out once you have a better feel for python.

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

#Section 2 : Real dataset.
#Iris Dataset
#The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. For each flower examined, its sepal length, sepal width, petal length and petal width are recorded (these are the features).

#The three classes are 'setosa', 'versicolor', 'virginica', represented by target variables 0, 1 and 2 respectively.

#Features
#We will use only the first two features (sepal length and sepal width) for classification.

# Some popular datasets are available in the packages. Use them for experimenting.
iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target
#Visualizing the data
for i in range(X.shape[0]) :
    row = X[i]
    if (y[i] == 0) :
        plt.scatter(row[0], row[1], s = 100, marker = '.', color = 'C1')
    elif (y[i] == 1) :
        plt.scatter(row[0], row[1], s = 100, marker = '.', color = 'C2')
    else :
        plt.scatter(row[0], row[1], s = 100, marker = '.', color = 'C3')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

#Choose only the first two classes
#(They are linearly separable)

selected_rows = [i for i in range(len(y)) if y[i] < 2]
X_new = X[selected_rows]
y_new = y[selected_rows]

for i in range(X_new.shape[0]) :
    row = X_new[i]
    if (y_new[i] == 0) :
        plt.scatter(row[0], row[1], s = 100, marker = '.', color = 'C1')
    elif (y_new[i] == 1) :
        plt.scatter(row[0], row[1], s = 100, marker = '.', color = 'C2')
    else :
        plt.scatter(row[0], row[1], s = 100, marker = '.', color = 'C3')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

#Binary classification, linearly separable
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X_new, y_new) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X_new[:, 0], X_new[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y_new, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

#Multi-class classification, not linearly separable
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

