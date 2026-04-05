# ==========================================
# CREDIT CARD FRAUD DETECTION SYSTEM
# WITH FILE SELECTOR
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# File dialog
import tkinter as tk
from tkinter import filedialog

# ==========================
# 1. SELECT DATASET FILE
# ==========================
root = tk.Tk()
root.withdraw()  # Hide main window

print("📂 Please select your dataset (CSV file)...")

file_path = filedialog.askopenfilename(
    title="Select Credit Card Dataset",
    filetypes=[("CSV Files", "*.csv")]
)

if not file_path:
    print("❌ No file selected. Exiting...")
    exit()

print(f"✅ Selected File: {file_path}")

# ==========================
# 2. LOAD DATASET
# ==========================
try:
    data = pd.read_csv(file_path)
    print("✅ Dataset Loaded Successfully!\n")
except Exception as e:
    print("❌ Error loading dataset:", e)
    exit()

# ==========================
# 3. BASIC INFO
# ==========================
print("📊 Dataset Shape:", data.shape)
print("\n🔍 First 5 Rows:\n", data.head())

# ==========================
# 4. CHECK REQUIRED COLUMN
# ==========================
if 'Class' not in data.columns:
    print("❌ Error: Dataset must contain 'Class' column (0 = Normal, 1 = Fraud)")
    exit()

# ==========================
# 5. CLASS DISTRIBUTION
# ==========================
print("\n⚖ Class Distribution:")
print(data['Class'].value_counts())

# ==========================
# 6. VISUALIZATION
# ==========================
plt.figure()
sns.countplot(x='Class', data=data)
plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Count")
plt.show()

# ==========================
# 7. SPLIT DATA
# ==========================
X = data.drop('Class', axis=1)
y = data['Class']

# ==========================
# 8. TRAIN-TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================
# 9. MODEL TRAINING
# ==========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\n✅ Model Training Completed!")

# ==========================
# 10. PREDICTION
# ==========================
y_pred = model.predict(X_test)

# ==========================
# 11. EVALUATION
# ==========================
print("\n📌 Accuracy:", accuracy_score(y_test, y_pred))

print("\n📌 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\n📌 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================
# 12. SAMPLE PREDICTION
# ==========================
sample = X_test.iloc[0].values.reshape(1, -1)
prediction = model.predict(sample)

print("\n🔎 Sample Transaction Prediction:")
if prediction[0] == 1:
    print("⚠ Fraudulent Transaction Detected")
else:
    print("✅ Normal Transaction")