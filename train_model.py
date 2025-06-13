import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("career_dataset.csv", header=0)

# Encode gender and part_time_job
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Part_Time_Job'] = df['Part_Time_Job'].map({'No': 0, 'Yes': 1})

# Encode hobbies flag
df['Hobbies_flag'] = df['Hobbies'].notnull().astype(int)

# Features and target
features = ['Gender', 'Part_Time_Job', 'Hobbies_flag', 'Weekly_Self_Study_Hours',
            'Math_Score', 'History_Score', 'Physics_Score', 'Chemistry_Score',
            'Biology_Score', 'English_Score', 'Geography_Score',
            'Total_Score', 'Average_Score']
X = df[features]
y = df['Career'].str.strip()  # remove extra spaces from career labels

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save class names
with open('class_names.pkl', 'wb') as f:
    pickle.dump(le.classes_.tolist(), f)

# Convert all columns to numeric and scale
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Define multiple models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
     
}

# Train and evaluate models
accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f" {name} Accuracy: {acc:.4f}")

# Select best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name} (Accuracy: {accuracies[best_model_name]:.2f})")

# Save best model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
