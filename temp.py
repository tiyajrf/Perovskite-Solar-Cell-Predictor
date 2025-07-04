import numpy as np
import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Set seed and number of samples
np.random.seed(42)
n_samples = 150

# Material options
a_site_materials = ["MA", "FA", "Cs"]
b_site_materials = ["Pb", "Sn", "Ge"]
c_site_materials = ["I", "Br", "Cl"]

# Create synthetic dataset
data = {
    "A_site_material": np.random.choice(a_site_materials, n_samples),
    "B_site_material": np.random.choice(b_site_materials, n_samples),
    "C_site_material": np.random.choice(c_site_materials, n_samples),
    "Humidity": np.round(np.random.uniform(30, 60, n_samples), 1),
    "Defect_density": np.round(np.random.uniform(1e15, 1e17, n_samples), -12),
    "Encapsulation": np.random.choice(["Yes", "No"], n_samples),
    "Temperature_stability": np.round(np.random.uniform(50, 100, n_samples), 1),
    "A_radius": np.round(np.random.uniform(1.6, 2.5, n_samples), 2),
    "B_radius": np.round(np.random.uniform(0.6, 1.2, n_samples), 2),
    "C_radius": np.round(np.random.uniform(1.3, 2.2, n_samples), 2),
    "A_electron_affinity": np.round(np.random.uniform(2.5, 4.5, n_samples), 2),
    "B_electron_affinity": np.round(np.random.uniform(3.0, 5.5, n_samples), 2),
    "C_electron_affinity": np.round(np.random.uniform(2.0, 4.0, n_samples), 2),
    "A_bandgap": np.round(np.random.uniform(1.4, 1.8, n_samples), 2),
    "B_bandgap": np.round(np.random.uniform(0.5, 1.2, n_samples), 2),
    "C_bandgap": np.round(np.random.uniform(2.0, 3.2, n_samples), 2),
}

# Compute synthetic stability (in days)
humidity = data["Humidity"]
defect_density = data["Defect_density"]
encapsulation_factor = np.array([1.2 if e == "Yes" else 0.8 for e in data["Encapsulation"]])
temp_stability = data["Temperature_stability"]
a_bg = data["A_bandgap"]
b_bg = data["B_bandgap"]
c_bg = data["C_bandgap"]

stability_days = (
    200
    + 1.5 * temp_stability
    - 0.05 * humidity
    - 1e-15 * defect_density
    + 10 * encapsulation_factor
    - 15 * (a_bg + b_bg + c_bg)
    + np.random.normal(0, 5, n_samples)
)

data["Stability_in_days"] = np.round(stability_days, 1)

# Convert to DataFrame and save
df = pd.DataFrame(data)
csv_filename = "perovskite_enhanced_features_dataset.csv"
df.to_csv(csv_filename, index=False)

# ---------------- MODEL TRAINING ----------------

# Features and target
features = list(df.columns)
features.remove("Stability_in_days")
target = "Stability_in_days"

# Define numerical and categorical features
categorical_features = ["A_site_material", "B_site_material", "C_site_material", "Encapsulation"]
numerical_features = list(set(features) - set(categorical_features))

# Impute missing values if any
imputer_num = SimpleImputer(strategy="mean")
df[numerical_features] = imputer_num.fit_transform(df[numerical_features])

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

# Final pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained and evaluated.\nR² Score: {r2:.4f}\nRMSE: {rmse:.4f}")

# Save model
model_filename = "stability_material_rf_enhanced.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(pipeline, f)

print(f"✅ Model pipeline saved as '{model_filename}'")
