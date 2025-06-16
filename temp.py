import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Step 0: Generate Synthetic Dataset and Save as CSV

np.random.seed(42)
n_samples = 150

a_site_materials = ["MA", "FA", "Cs"]
b_site_materials = ["Pb", "Sn", "Ge"]
c_site_materials = ["I", "Br", "Cl"]

data = {
    "A_site_material": np.random.choice(a_site_materials, n_samples),
    "B_site_material": np.random.choice(b_site_materials, n_samples),
    "C_site_material": np.random.choice(c_site_materials, n_samples),
    "Perovskite_deposition_synthesis_atmosphere_relative_humidity": np.round(np.random.uniform(30, 60, n_samples), 1),
    "Perovskite_band_gap": np.round(np.random.uniform(1.45, 1.65, n_samples), 2),
}

# ✅ Improved Stability score formula with interaction term
bandgap = data["Perovskite_band_gap"]
humidity = data["Perovskite_deposition_synthesis_atmosphere_relative_humidity"]
interaction = np.array(bandgap) * np.array(humidity)

stability_score = (
    65
    - 0.25 * humidity
    - 8.5 * bandgap
    - 0.02 * interaction
    + np.random.normal(0, 2.2, n_samples)
)

data["Stability_measured"] = np.round(stability_score, 1)

df = pd.DataFrame(data)

csv_filename = "perovskite_stability_dataset_improved.csv"
df.to_csv(csv_filename, index=False)
print(f"✅ CSV dataset saved as '{csv_filename}'")

# Step 1: Load dataset from CSV
data = pd.read_csv(csv_filename)

# Step 2: Define features and target
features = [
    "A_site_material",
    "B_site_material",
    "C_site_material",
    "Perovskite_deposition_synthesis_atmosphere_relative_humidity",
    "Perovskite_band_gap",
    "Bandgap_Humidity_Interaction"
]
target = "Stability_measured"

# Step 3: Add interaction feature
data["Bandgap_Humidity_Interaction"] = data["Perovskite_band_gap"] * data["Perovskite_deposition_synthesis_atmosphere_relative_humidity"]

# Step 4: Handle missing values (if any)
categorical_features = ["A_site_material", "B_site_material", "C_site_material"]
numerical_features = [
    "Perovskite_deposition_synthesis_atmosphere_relative_humidity",
    "Perovskite_band_gap",
    "Bandgap_Humidity_Interaction"
]

imputer_num = SimpleImputer(strategy="mean")
data[numerical_features] = imputer_num.fit_transform(data[numerical_features])

# Step 5: Split into train and test
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build preprocessing + model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ✅ Improved model settings
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Step 7: Train the model
pipeline.fit(X_train, y_train)

# Step 8: Evaluate model
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Performance:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Step 9: Save the trained pipeline with a new .pkl filename
model_filename = "stability_material_rf_improved.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(pipeline, f)

print(f"✅ Model pipeline saved as '{model_filename}' at {os.path.abspath(model_filename)}")