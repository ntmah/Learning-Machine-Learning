import numpy as np 
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
np.random.seed(7)
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target 
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size = 50)
model = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy of 10NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))