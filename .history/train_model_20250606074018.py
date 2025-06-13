# train_model.py

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load your dataset (replace with your actual file)
data = pd.read_csv("career_dataset.csv")

# Preprocess
data['gender'] = data['gender'].map({'male': 0, 'female': 1})
data['part_time_job'] = data['part_time_job'].astype(int)
data['hobbies'] = data['hobbies'].astype(int)

X = data.drop(columns=['career'])  # target
y = data['career']

# Save class names
class_names = y.unique().tolist()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(class_names, open("class_names.pkl", "wb"))
