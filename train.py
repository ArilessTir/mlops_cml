from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

## Import data
X,y = load_diabetes(return_X_y=True, as_frame=True)

## Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## Fit

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_true=y_test, y_pred= y_pred)


## Metrics
file = open("metrics.txt", "w")
file.write(f'Mean Squared Error: {mse:.2f}')

## Plots
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=3, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calculate mean and standard deviation of training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color="blue", marker="o", markersize=5, label="Training score")
plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15,
    color="blue"
)
plt.plot(train_sizes, test_mean, color="green", linestyle="--", marker="s", markersize=5, label="Validation score")
plt.fill_between(
    train_sizes,
    test_mean - test_std,
    test_mean + test_std,
    alpha=0.15,
    color="green"
)

# Customize the plot
plt.title("Learning Curve")
plt.xlabel("Training Size")
plt.ylabel("Score")
plt.legend(loc="best")
plt.grid(True)
plt.savefig('learning_curve.png')

