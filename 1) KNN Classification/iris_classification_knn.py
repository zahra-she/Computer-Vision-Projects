"""
KNN Classification on Iris Dataset
----------------------------------

Steps:
1. Load the dataset
2. Split the dataset into training and testing sets
3. Train a KNN classifier
4. Evaluate the model with accuracy score

Author: [zahra-she]
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()

# Extract features and labels from the dataset
features = iris.data  # Sepal length, sepal width, petal length, petal width
labels = iris.target  # 0: Setosa, 1: Versicolor, 2: Virginica

# Split the dataset into training (80%) and testing (20%) sets
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Create and configure the KNN classifier with 7 neighbors
model = KNeighborsClassifier(n_neighbors=7)

# Train the model on the training data
model.fit(x_train, y_train)

# Predict the labels of the test set
y_pred = model.predict(x_test)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("accuracy: {:.2f}".format(accuracy * 100))
