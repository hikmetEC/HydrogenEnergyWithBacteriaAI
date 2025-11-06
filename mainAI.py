import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 1️⃣ Generate simulated data
np.random.seed(42)
n = 500

sicaklik = np.random.uniform(25, 40, n)
isik = np.random.uniform(100, 1000, n)
pH = np.random.uniform(6.0, 8.5, n)
besin = np.random.uniform(0.5, 1.5, n)

hidrojen = (
    0.3 * sicaklik +
    0.4 * isik / 1000 +
    0.5 * besin -
    0.2 * abs(pH - 7.2) +
    np.random.normal(0, 0.5, n)
)

# Combine features
X = np.column_stack([sicaklik, isik, pH, besin])
y = hidrojen

# 2️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4️⃣ Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.3f}")

# 5️⃣ Visualize
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("True Production (simulated)")
plt.ylabel("Predicted Production")
plt.title("AI Model - Hydrogen Production (Simulation)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# 6️⃣ Example prediction
example = np.array([[32, 700, 7.1, 1.0]])
prediction = model.predict(example)[0]
print(f"Predicted hydrogen output (simulation): {prediction:.2f}")
