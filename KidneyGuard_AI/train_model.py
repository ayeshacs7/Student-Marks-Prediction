# ============================================
# KidneyGuard AI - Train Machine Learning Model
# ============================================

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load Dataset
data = pd.read_csv("dataset/kidney.csv")

# Show First 5 Rows
print("First 5 Rows of Dataset:\n")
print(data.head())

# Show Dataset Information
print("\n==============================")
print("Dataset Information")
print("==============================")
print(data.info())

# Show Shape
print("\nDataset Shape:", data.shape)

# Show Missing Values
print("\nMissing Values:\n")
print(data.isnull().sum())
# ====================================
# Column Names
# ====================================

print("\n==============================")
print("Column Names")
print("==============================")

for column in data.columns:
    print(column)

# ====================================
# Data Types
# ====================================

print("\n==============================")
print("Data Types")
print("==============================")

print(data.dtypes)

# ====================================
# Target Distribution
# ====================================

print("\n==============================")
print("Target Distribution")
print("==============================")

print(data.iloc[:, -1].value_counts())
# ====================================
# Missing Values
# ====================================

print("\n==============================")
print("Missing Values")
print("==============================")

print(data.isnull().sum())
# ====================================
# Missing Values
# ====================================

print("\n==============================")
print("Missing Values")
print("==============================")

print(data.isnull().sum())
# ====================================
# Split Features and Target
# ====================================

X = data.drop("Class", axis=1)
y = data["Class"]

# ====================================
# Train Test Split
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ====================================
# Train Random Forest Model
# ====================================

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# ====================================
# Prediction
# ====================================

y_pred = model.predict(X_test)

# ====================================
# Accuracy
# ====================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# ====================================
# Classification Report
# ====================================

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ====================================
# Save Model
# ====================================

joblib.dump(model, "models/model.pkl")

print("\n✅ Model Saved Successfully!")
print("\nFirst 10 Rows")
print(data.head(10))
print("\nUnique values in Class:")
print(data["Class"].unique())
print("\nTarget Distribution")
print(data["Class"].value_counts())
print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

print("\nPredicted Values:")
print(y_pred[:20])

print("\nActual Values:")
print(y_test.values[:20])