 # app.py
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

class_names = ['Engineer', 'Doctor', 'Artist', 'Business Owner', 'Real Estate Developer', 'Unknown']  # Example classes

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

@app.route('/')
def home():
    return render_template('recommend.html')

@app.route('/pred', methods=['POST'])
def Recommendations():
    if request.method == 'POST':
        gender = request.form['gender']
        part_time_job = request.form['part_time_job']
        hobbies = request.form.get('hobbies')
        intermediate_score = safe_float(request.form.get('intermediate_score'))
        history_score = safe_float(request.form.get('history_score'))
        physics_score = safe_float(request.form.get('physics_score'))
        chemistry_score = safe_float(request.form.get('chemistry_score'))
        biology_score = safe_float(request.form.get('biology_score'))
        english_score = safe_float(request.form.get('english_score'))
        geography_score = safe_float(request.form.get('geography_score'))
        total_score = safe_float(request.form.get('total_score'))
        average_score = safe_float(request.form.get('average_score'))

        # Encode categorical data
        gender_encoded = 1 if gender.lower() == 'male' else 0
        part_time_job_encoded = 1 if part_time_job.lower() == 'true' else 0
        hobby_list = ['reading', 'sports', 'music', 'painting', 'gaming', 'traveling', 'coding', 'dancing']
        hobbies_encoded = [1 if hobbies == hobby else 0 for hobby in hobby_list]

        # Feature Array
        feature_array = np.array([[gender_encoded, part_time_job_encoded] + hobbies_encoded + [
            intermediate_score, history_score, physics_score,
            chemistry_score, biology_score, english_score,
            geography_score, total_score, average_score]])

        # Prediction probabilities
        probabilities = model.predict_proba(feature_array)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]  # Top 3 recommendations
        recommendations = [(class_names[i], round(probabilities[i]*100, 2)) for i in top_indices]

        return render_template('recommend.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
