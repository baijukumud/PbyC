import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Create a simple dataset
# Let's predict 'y' from 'x' with a linear relationship
np.random.seed(42)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)  # y = 4 + 3x + noise

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 3. Fit a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Interpret Coefficients
print("Intercept:", model.intercept_[0])
print("Coefficient:", model.coef_[0][0])

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate Performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):",r2)
