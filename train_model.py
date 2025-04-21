import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load dataset
df = pd.read_csv("WineQT1.csv")
df.dropna(inplace=True)
df["quality_binary"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

# Features and target
X = df.drop(["quality", "quality_binary"], axis=1)
y = df["quality_binary"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model (K = 5 default)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("✅ Model, scaler, and feature names saved.")
