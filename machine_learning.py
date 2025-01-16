import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    "Hours Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Exam Score": [50, 55, 65, 70, 75, 80, 85, 88, 93, 95]
}
df = pd.DataFrame(data)

plt.scatter(df["Hours Studied"], df["Exam Score"], color="blue")
plt.title("Hours Studied vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.show()

X = df[["Hours Studied"]]  
y = df["Exam Score"]       


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

plt.scatter(X_test, y_test, color="red", label="Actual")
plt.scatter(X_test, y_pred, color="green", label="Predicted")
plt.plot(X_test, y_pred, color="blue", label="Regression Line")
plt.legend()
plt.title("Actual vs Predicted")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.show()

hours = float(input("Enter hours studied: "))
predicted_score = model.predict([[hours]])
print(f"Predicted Exam Score: {predicted_score[0]:.2f}")
