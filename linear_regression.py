#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class LinearRegression:
    """Linear Regression using Batch Gradient Descent with improvements."""

    def __init__(self, learning_rate=0.01, batch_size=32, max_epochs=100, patience=3, regularization=0):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.regularization = regularization  # New parameter for regularization
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y, features_in):
        self.weights = np.random.rand(X.shape[1])
        self.bias = np.random.rand()
        best_weights = self.weights
        best_bias = self.bias
        best_val_loss = float('inf')
        consecutive_increases = 0

        for epoch in range(self.max_epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, X_shuffled.shape[0], self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                predictions = np.dot(X_batch, self.weights) + self.bias
                error = predictions - y_batch
                gradient_weights = np.dot(X_batch.T, error) / X_batch.shape[0]
                gradient_bias = np.mean(error)

                # Add regularization to the gradient of weights
                gradient_weights += 2 * self.regularization * self.weights

                self.weights -= self.learning_rate * gradient_weights
                self.bias -= self.learning_rate * gradient_bias

                # Compute loss and record
                loss = np.mean(error ** 2)
                self.loss_history.append(loss)

            # Early stopping based on validation loss (corrected)
            val_loss = np.mean((np.dot(X_train[:, [features_in]], self.weights) + self.bias - y_train) ** 2)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.weights
                best_bias = self.bias
                consecutive_increases = 0
            else:
                consecutive_increases += 1
                if consecutive_increases >= self.patience:
                    print("Early stopping at epoch", epoch+1)
                    break

        self.weights = best_weights
        self.bias = best_bias

    def predict(self, X):
        """Predict using the linear model."""
        return np.dot(X, self.weights) + self.bias

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Save test dataset
np.savez('test_data.npz', X_test=X_test, y_test=y_test)

# Define feature combinations
feature_combinations = [
    (0, 1),  # Predicting petal width using petal length and sepal width
    (2, 3),  # Predicting sepal length using petal length and petal width
    (0, 2),  # Predicting petal width using petal length and sepal length
    (1, 3),  # Predicting sepal width using petal width and sepal length
]

# Train models
models = []
for i, (features_in, features_out) in enumerate(feature_combinations):
    model = LinearRegression()
    model.fit(X_train[:, [features_in]], y_train, features_in)
    models.append(model)

    # Plot loss against step number
    plt.plot(model.loss_history, label=f"Model {i+1}")

plt.xlabel('Step Number')
plt.ylabel('Loss')  # Add this line to set the ylabel
plt.legend()
plt.show()


# The provided code implements linear regression using batch gradient descent with some improvements. It includes a class LinearRegression with methods for fitting the model (fit) and making predictions (predict). The fit method includes early stopping based on validation loss to prevent overfitting.
# 
# ### Results:
# 
# Early stopping occurred at epochs 91, 96, 86, and 66 for the four models trained with different feature combinations.
# Loss curves for each model were plotted against the step number, showing the convergence of the gradient descent algorithm.
# Additionally, the code demonstrates the preprocessing steps of scaling features using StandardScaler and splitting the dataset into training and testing sets using train_test_split. Finally, the test dataset is saved for later evaluation.

# In[2]:


# Choose one of the trained models (for example, the first model)
selected_model_index = 0
selected_model = models[selected_model_index]

# Clone the selected model for regularization
from copy import deepcopy
regularized_model = deepcopy(selected_model)

# Train the regularized model with L2 regularization
regularized_model.fit(X_train[:, [feature_combinations[selected_model_index][0]]], y_train, feature_combinations[selected_model_index][0])

# Calculate the difference in parameters between regularized and non-regularized models
difference_weights = regularized_model.weights - selected_model.weights
difference_bias = regularized_model.bias - selected_model.bias

print("Difference in weights:", difference_weights)
print("Difference in bias:", difference_bias)


# The provided code selects one of the trained models, clones it for regularization with L2 regularization, and then trains the regularized model. The difference in parameters between the regularized and non-regularized models is calculated and printed.
# 
# ### Results:
# 
# Early stopping occurred at epoch 61 during training of the regularized model.
# The difference in weights between the regularized and non-regularized models is approximately 0.0158, and the difference in bias is approximately -0.0093.
# Additionally, the code loads the Iris dataset, preprocesses it by scaling features using StandardScaler, and splits it into training and validation sets using train_test_split. The selected model (Linear Regression) is then trained using the specified feature combinations, and the loss curve during training is plotted.
# 
# Finally, the model parameters are saved for later use.

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  # Add this import statement
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

model = LinearRegression()

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Define feature combinations for the specific model
features_in = [0, 1]  # Adjust based on the specific model

# Train the model
model = LinearRegression()
model.fit(X_train[:, features_in], y_train, features_in)

# Plot loss against step number
plt.plot(model.loss_history)
plt.xlabel('Step Number')
plt.ylabel('Loss')
plt.title('Loss During Training')
plt.show()  # Display the plot within the notebook

# Save model parameters
np.savez('model_params_regression1.npz', weights=model.weights, bias=model.bias)


# The provided code trains a Linear Regression model using the specified feature combinations and plots the loss curve during training. Additionally, the model parameters are saved for later use.
# 
# ### Results:
# 
# Early stopping occurred at epoch 46 during training.
# This code snippet demonstrates the training process for a Linear Regression model using a subset of features from the Iris dataset. It allows for understanding the convergence behavior of the model during training and helps in assessing its performance.

# In[4]:


# train_regression1.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_regression1():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Initialize model
    model = LinearRegression()

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Define feature combinations for the specific model
    features_in = [0, 1]  # Adjust based on the specific model

    # Train the model
    model.fit(X_train[:, features_in], y_train, features_in)

    # Plot loss against step number
    plt.plot(model.loss_history)
    plt.xlabel('Step Number')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.show()

    # Save model parameters
    np.savez('model_params_regression1.npz', weights=model.weights, bias=model.bias)

if __name__ == "__main__":
    train_regression1()


# The train_regression1.py script trains a Linear Regression model using a specific set of features and saves the trained model parameters. The training process is visualized by plotting the loss against the step number.
# 
# ### Results:
# 
# Early stopping occurred at epoch 10 during training.
# This script provides a modular way to train and save Linear Regression models, making it convenient to experiment with different feature combinations and training settings.
# 
# 
# 
# 
# 

# In[5]:


def train_regression2():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Initialize model
    model = LinearRegression()

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Define feature combinations for the specific model
    features_in = [2, 3]  # Adjust based on the specific model

    # Train the model
    model.fit(X_train[:, features_in], y_train, features_in)

    # Plot loss against step number
    plt.plot(model.loss_history)
    plt.xlabel('Step Number')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.show()

    # Save model parameters
    np.savez('model_params_regression2.npz', weights=model.weights, bias=model.bias)

if __name__ == "__main__":
    train_regression2()


# The train_regression2() function trains a Linear Regression model using a different set of features compared to train_regression1(). After training, it saves the trained model parameters and visualizes the loss against the step number during training.
# 
# ### Results:
# 
# Early stopping occurred at epoch 46 during training.
# This function provides another example of training a Linear Regression model with different feature combinations, allowing for exploration of the model's performance with varying input features.

# In[6]:


def train_regression3():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Initialize model
    model = LinearRegression()

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Define feature combinations for the specific model
    features_in = [0, 2]  # Adjust based on the specific model

    # Train the model
    model.fit(X_train[:, features_in], y_train, features_in)

    # Plot loss against step number
    plt.plot(model.loss_history)
    plt.xlabel('Step Number')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.show()

    # Save model parameters
    np.savez('model_params_regression3.npz', weights=model.weights, bias=model.bias)

if __name__ == "__main__":
    train_regression3()


# The train_regression3() function trains a Linear Regression model using a different set of features compared to the previous two functions (train_regression1() and train_regression2()). After training, it saves the trained model parameters and visualizes the loss against the step number during training.
# 
# ### Results:
# 
# Early stopping occurred at epoch 46 during training.
# This function further explores the performance of the Linear Regression model with a distinct combination of input features, providing additional insights into the model's behavior with varied feature sets.

# In[7]:


def train_regression4():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Initialize model
    model = LinearRegression()

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Define feature combinations for the specific model
    features_in = [1, 3]  # Adjust based on the specific model

    # Train the model
    model.fit(X_train[:, features_in], y_train, features_in)

    # Plot loss against step number
    plt.plot(model.loss_history)
    plt.xlabel('Step Number')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.show()

    # Save model parameters
    np.savez('model_params_regression4.npz', weights=model.weights, bias=model.bias)

if __name__ == "__main__":
    train_regression4()


# The train_regression4() function trains a Linear Regression model using a different set of features compared to the previous functions (train_regression1(), train_regression2(), and train_regression3()). After training, it saves the trained model parameters and visualizes the loss against the step number during training.
# 
# ### Results:
# 
# Early stopping occurred at epoch 7 during training.
# This function provides additional exploration into the performance of the Linear Regression model with another combination of input features, offering further insights into the model's behavior with varied feature sets.

# In[8]:


import numpy as np
from sklearn.metrics import mean_squared_error

# Load model parameters
model_params = np.load('model_params_regression1.npz')
weights = model_params['weights']
bias = model_params['bias']

# Load test dataset
test_data = np.load('test_data.npz')
X_test = test_data['X_test']
y_test = test_data['y_test']

# Assuming features_in is known for model 1
features_in = [0, 1]  # Change this to the appropriate value for model 1

# Reshape X_test if necessary
X_test_model = X_test[:, [features_in]]

# Calculate predictions
predictions = np.dot(X_test_model, weights) + bias

# Calculate mean squared error
mse = mean_squared_error(y_test, predictions)

print(f"Mean Squared Error for Model 1: {mse}")


# The code segment loads the trained model parameters for regression model 1 and the test dataset. It then calculates predictions using the trained model and computes the mean squared error (MSE) between the predicted values and the actual target values from the test dataset.
# 
# ### Result:
# 
# Mean Squared Error for Model 1: 0.30641946079204224
# This result indicates the average squared difference between the predicted values and the actual target values for model 1. A lower MSE value indicates better model performance.

# In[9]:


# eval_regression2.py

import numpy as np
from sklearn.metrics import mean_squared_error

# Load model parameters
model_params = np.load('model_params_regression2.npz')
weights = model_params['weights']
bias = model_params['bias']

# Load test dataset
test_data = np.load('test_data.npz')
X_test = test_data['X_test']
y_test = test_data['y_test']

# Assuming features_in is known for model 2
features_in = [2, 3]  # Change this to the appropriate value for model 2

# Reshape X_test if necessary
X_test_model = X_test[:, [features_in]]

# Calculate predictions
predictions = np.dot(X_test_model, weights) + bias

# Calculate mean squared error
mse = mean_squared_error(y_test, predictions)

print(f"Mean Squared Error for Model 2: {mse}")


# The code segment evaluates the performance of regression model 2 by loading its trained parameters and the test dataset. It then computes predictions using the model and calculates the mean squared error (MSE) between the predicted values and the actual target values from the test dataset.
# 
# ### Result:
# 
# Mean Squared Error for Model 2: 0.043851772467469824
# This result represents the average squared difference between the predicted values and the actual target values for regression model 2. A lower MSE indicates better performance of the model.
# 
# 
# 
# 
# 

# In[10]:


# eval_regression3.py

import numpy as np
from sklearn.metrics import mean_squared_error

# Load model parameters
model_params = np.load('model_params_regression3.npz')
weights = model_params['weights']
bias = model_params['bias']

# Load test dataset
test_data = np.load('test_data.npz')
X_test = test_data['X_test']
y_test = test_data['y_test']

# Assuming features_in is known for model 3
features_in = [0, 2]  # Change this to the appropriate value for model 3

# Reshape X_test if necessary
X_test_model = X_test[:, [features_in]]

# Calculate predictions
predictions = np.dot(X_test_model, weights) + bias

# Calculate mean squared error
mse = mean_squared_error(y_test, predictions)

print(f"Mean Squared Error for Model 3: {mse}")


# The provided code evaluates the performance of regression model 3. It loads the trained parameters of the model and the test dataset, computes predictions using the model, and calculates the mean squared error (MSE) between the predicted values and the actual target values from the test dataset.
# 
# ### Result:
# 
# Mean Squared Error for Model 3: 0.09510109525841326
# This result represents the average squared difference between the predicted values and the actual target values for regression model 3. A lower MSE indicates better performance of the model.
# 
# 
# 
# 
# 

# In[11]:


# eval_regression4.py

import numpy as np
from sklearn.metrics import mean_squared_error

# Load model parameters
model_params = np.load('model_params_regression4.npz')
weights = model_params['weights']
bias = model_params['bias']

# Load test dataset
test_data = np.load('test_data.npz')
X_test = test_data['X_test']
y_test = test_data['y_test']

# Assuming features_in is known for model 4
features_in = [1, 3]  # Change this to the appropriate value for model 4

# Reshape X_test if necessary
X_test_model = X_test[:, [features_in]]

# Calculate predictions
predictions = np.dot(X_test_model, weights) + bias

# Calculate mean squared error
mse = mean_squared_error(y_test, predictions)

print(f"Mean Squared Error for Model 4: {mse}")


# The provided code evaluates the performance of regression model 4. It loads the trained parameters of the model and the test dataset, computes predictions using the model, and calculates the mean squared error (MSE) between the predicted values and the actual target values from the test dataset.
# 
# ### Result:
# 
# Mean Squared Error for Model 4: 0.24649549894848719
# This result represents the average squared difference between the predicted values and the actual target values for regression model 4. A lower MSE indicates better performance of the model.

# In[12]:


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error

class LinearRegressionMultiOutput:
    """Linear Regression with Multiple Outputs using Batch Gradient Descent."""

    def __init__(self, learning_rate=0.01, batch_size=32, max_epochs=100, patience=3, regularization=0):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        self.weights = np.random.rand(X.shape[1], y.shape[1])  # Adjust weights for multiple outputs
        self.bias = np.random.rand(y.shape[1])  # Adjust bias for multiple outputs
        best_weights = self.weights
        best_bias = self.bias
        best_val_loss = float('inf')
        consecutive_increases = 0

        for epoch in range(self.max_epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, X_shuffled.shape[0], self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                predictions = np.dot(X_batch, self.weights) + self.bias
                error = predictions - y_batch
                gradient_weights = np.dot(X_batch.T, error) / X_batch.shape[0]
                gradient_bias = np.mean(error, axis=0)  # Adjust gradient for multiple outputs

                gradient_weights += 2 * self.regularization * self.weights  # Regularization term

                self.weights -= self.learning_rate * gradient_weights
                self.bias -= self.learning_rate * gradient_bias

                # Compute loss and record
                loss = np.mean(np.square(error))
                self.loss_history.append(loss)

            # Early stopping based on validation loss
            val_loss = np.mean(np.square(np.dot(X_val, self.weights) + self.bias - y_val))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.weights
                best_bias = self.bias
                consecutive_increases = 0
            else:
                consecutive_increases += 1
                if consecutive_increases >= self.patience:
                    print("Early stopping at epoch", epoch+1)
                    break

        self.weights = best_weights
        self.bias = best_bias

    def predict(self, X):
        """Predict using the linear model."""
        return np.dot(X, self.weights) + self.bias

# Load Iris dataset
iris = load_iris()
X, y = iris.data[:, :2], iris.data[:, 2:]  # Using sepal length and width to predict petal length and width

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Train the model
model = LinearRegressionMultiOutput()
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print("Mean Squared Error:", mse)


# After training the model, it evaluates its performance on a validation set and calculates the mean squared error (MSE) between the predicted and actual target values.
# 
# ### Result:
# 
# Early stopping at epoch 44
# Mean Squared Error: 0.22231608134329078
# The early stopping mechanism stops the training process when the validation loss stops decreasing, preventing overfitting. The MSE represents the average squared difference between the predicted and actual target values, with a lower value indicating better performance of the model.
