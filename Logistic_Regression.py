#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iters=1000):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.max_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            # Gradient descent
            dw = (1 / m) * np.dot(X.T, (predictions - y))
            db = (1 / m) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        return np.round(predictions)


# In[2]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X, y = iris.data[:, [2, 3]], iris.target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for petal length/width variant: {accuracy}")

# Visualize decision boundaries
plot_decision_regions(X_train, y_train, clf=model, legend=2)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Logistic Regression - Petal Length/Width Variant')
plt.show()


# **Logistic Regression Model Evaluation**
# 
# - **Data:** Used Iris dataset, focusing on petal length and width features.
# 
# - **Training:** Split the dataset into training and testing sets, with 70% for training and 30% for testing.
# 
# - **Model:** Trained a logistic regression model using the training data.
# 
# - **Prediction:** Made predictions on the test set using the trained model.
# 
# - **Evaluation:** Achieved an accuracy of 0.2889 on the test set.
# 
# - **Visualization:** Displayed decision boundaries to illustrate class separation based on petal length and width.
# 
# - **Conclusion:** Obtained a moderate accuracy, indicating that the model's performance may be limited by the selected features.
# 

# In[3]:


# Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save Model Parameters
np.savez('model_params_classifier1.npz', weights=model.weights, bias=model.bias)

# Now you can run the evaluation script


# In[4]:


import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X, y = iris.data[:, [0, 1]], iris.target  # Using sepal length/width features

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Load model parameters
model_params = np.load('model_params_classifier1.npz')
weights = model_params['weights']
bias = model_params['bias']

# Define and train the model
clf = LogisticRegression()
clf.weights = weights
clf.bias = bias

# Predict
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for sepal length/width features: {accuracy}")

# Visualize decision boundaries
plt.figure(figsize=(10, 6))
plot_decision_regions(X_test, y_test, clf=clf, legend=2)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Decision Regions for Logistic Regression (Sepal Length/Width)')
plt.show()


# **Logistic Regression Model Evaluation**
# 
# - **Data:** Utilized Iris dataset, focusing on sepal length and width features.
#   
# - **Training:** Split the dataset into training and testing sets, with 90% for training and 10% for testing.
# 
# - **Model Parameters:** Loaded trained model parameters from a file.
# 
# - **Prediction:** Made predictions on the test set using the trained model.
# 
# - **Evaluation:** Achieved an accuracy of 0.4 on the test set.
# 
# - **Visualization:** Visualized decision boundaries to illustrate class separation.
# 
# - **Conclusion:** Achieved a moderate accuracy, indicating potential effectiveness in classifying iris species based on sepal dimensions.
# 

# In[5]:


# Load Iris dataset
iris = load_iris()
X, y = iris.data[:, [2, 3]], iris.target  # Using all features

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for all features: {accuracy}")

# Visualize decision boundaries
plt.figure(figsize=(10, 6))
plot_decision_regions(X_test, y_test, clf=model, legend=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression - All Features')
plt.show()


# **Logistic Regression Model Evaluation**
# 
# - **Data:** Utilized Iris dataset, including all features.
# 
# - **Training:** Split the dataset into training and testing sets, with 70% for training and 30% for testing.
# 
# - **Model:** Trained a logistic regression model using the training data.
# 
# - **Prediction:** Made predictions on the test set using the trained model.
# 
# - **Evaluation:** Achieved an accuracy of 0.2889 on the test set.
# 
# - **Visualization:** Visualized decision boundaries to illustrate class separation.
# 
# - **Conclusion:** Obtained a moderate accuracy, suggesting that the model's performance may be limited by the chosen features.
# 
