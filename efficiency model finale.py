# ========================== #
# Step 1: Import Libraries
# ========================== #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# ========================== #
# Step 2: Create & Save Synthetic Dataset
# ========================== #

htl_materials = ['Spiro-OMeTAD', 'PTAA', 'NiOx']
etl_materials = ['TiO2', 'SnO2', 'PCBM']
np.random.seed(42)
num_samples = 150

synthetic_efficiency_data = pd.DataFrame({
    "Perovskite_thickness_nm": np.random.uniform(300, 700, num_samples),
    "Perovskite_deposition_temperature_C": np.random.uniform(80, 150, num_samples),
    "Perovskite_annealing_time_min": np.random.uniform(10, 60, num_samples),
    "Perovskite_band_gap_eV": np.random.uniform(1.45, 1.65, num_samples),
    "HTL_material": np.random.choice(htl_materials, num_samples),
    "HTL_thickness_nm": np.random.uniform(10, 100, num_samples),
    "ETL_material": np.random.choice(etl_materials, num_samples),
    "ETL_thickness_nm": np.random.uniform(10, 100, num_samples),
    "Efficiency_measured": np.random.uniform(12.0, 22.0, num_samples)
})

# Save to CSV
csv_path = "internship_perovskite_efficiency_data.csv"
synthetic_efficiency_data.to_csv(csv_path, index=False)
print(f"✅ Synthetic dataset saved as '{csv_path}'")

# ========================== #
# Step 3: Load Dataset
# ========================== #

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Error: File '{csv_path}' not found.")

data = pd.read_csv(csv_path)
print("✅ Dataset loaded successfully!")
print(data.head())

# ========================== #
# Step 4: Define Features & Target
# ========================== #

features = [
    "Perovskite_thickness_nm",
    "Perovskite_deposition_temperature_C",
    "Perovskite_annealing_time_min",
    "Perovskite_band_gap_eV",
    "HTL_material",
    "HTL_thickness_nm",
    "ETL_material",
    "ETL_thickness_nm"
]

target = "Efficiency_measured"
X = data[features]
y = data[target]

# ========================== #
# Step 5: Preprocessing
# ========================== #

categorical_features = ["HTL_material", "ETL_material"]
numerical_features = list(set(features) - set(categorical_features))

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ========================== #
# Step 6: Train/Test Split
# ========================== #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================== #
# Step 7: Pipeline & Training
# ========================== #
model = RandomForestRegressor(n_estimators=200, random_state=42)
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

# ========================== #
# Step 8: Evaluation
# ========================== #
y_pred = pipeline.predict(X_test)
print("\n📊 Model Performance:")
print("R² Score:", round(r2_score(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))

# ========================== #
# Step 9: Save the Model
# ========================== #
model_path = "efficiency_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)
print(f"\n✅ Model pipeline saved as '{model_path}'")

# ========================== #
# Step 10: Plot Performance
# ========================== #
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("True Efficiency (%)")
plt.ylabel("Predicted Efficiency (%)")
plt.title("Efficiency Prediction Performance")
plt.tight_layout()
plt.show()

