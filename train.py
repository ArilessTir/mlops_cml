from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

## Import data
X,y = load_diabetes(return_X_y=True, as_frame=True)

## Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## Fit

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_true=y_test, y_pred= y_pred)

print(mse)