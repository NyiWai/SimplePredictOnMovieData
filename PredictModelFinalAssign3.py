import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings

data = pd.read_csv('CleanData.csv')
X = pd.DataFrame(data, columns=['ProBudget'])
y = pd.DataFrame(data, columns=['Gross'])

model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

coefficients = model.coef_[0]
intercept = model.intercept_

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\n\nRoot Mean Squared Error: {rmse}")

r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

print("\nModel Coefficients:")
for i, coef in enumerate(coefficients):
    print(f"Coefficient {i+1}: {coef}")

print(f"\nIntercept: {intercept}")

# Visualization
plt.scatter(X_test, y_test, color='black', label='Actual Data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Linear Regression Model')
plt.xlabel('ProBudget')
plt.ylabel('Gross')
plt.title('Linear Regression Model Visualization')
plt.legend()
plt.show()
# Prediction
# Accept user input for relevant features
user_input = float(input("\nEnter the value of ProBudget for prediction: "))
user_input = np.array([[user_input]])  # Reshape input for prediction
# Make predictions using the trained model
prediction = model.predict(user_input)
warnings.filterwarnings('ignore')

# Display the predicted output
print(f"\nPredicted Gross: {prediction[0]}")
