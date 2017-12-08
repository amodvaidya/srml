import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import Perceptron
from lib import plot_decision_regions as pdr



# df = pd.read_csv('zzzhttps://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df = pd.read_csv('iris.data', header=None)
print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y=='Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values
plt.scatter(X[:50,0], X[:50,1],color='red', marker='o', label='sentosa')
plt.scatter(X[50:100,0], X[50:100,1],color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()
print("Done plotting data")
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
print("Done learning")
pdr(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()


