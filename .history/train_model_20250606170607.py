import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("career_dataset.csv")

# Encode gender and part_time_job
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Part_Time_Job'] = df['Part_Time_Job'].map({'No': 0, 'Yes': 1})

# Encode hobbies flag (1 if any hobby is selected)
df['Hobbies_flag'] = df['hobbies'].notnull().astype(int)

# Select only the columns you actually use in app.py
features = ['gender', 'part_time_job', 'hobbies_flag', 'weekly_self_study_hours',
            'math_score', 'history_score', 'physics_score', 'chemistry_score',
            'biology_score', 'english_score', 'geography_score',
            'total_score', 'average_score']

X = df[features]
y = df['career']  # Replace with your actual target column

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save class names
class_names = le.classes_.tolist()
with open('class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_scaled, y_encoded)

# Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
