from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model components
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
class_names = pickle.load(open('class_names.pkl', 'rb'))  # List of class labels

# Predict function
def Recommendations(gender, part_time_job, hobbies, weekly_self_study_hours,
                    math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):
    
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    hobbies_encoded = 1 if hobbies else 0

    # Feature array
    feature_array = np.array([[gender_encoded, part_time_job_encoded, hobbies_encoded,
                               weekly_self_study_hours, math_score, history_score, physics_score,
                               chemistry_score, biology_score, english_score, geography_score,
                               total_score, average_score]])
    
    # Scaling
    scaled_features = scaler.transform(feature_array)

    # Predict
    probabilities = model.predict_proba(scaled_features)
    top_classes_idx = np.argsort(-probabilities[0])[:3]
    top_classes = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]
    
    return top_classes

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

@app.route('/pred', methods=['POST', 'GET'])
def pred():
    if request.method == 'POST':
        gender = request.form['gender']
        part_time_job = request.form['part_time_job'] == 'true'
        hobbies = request.form.getlist('hobbies')
        weekly_self_study_hours = int(request.form['weekly_self_study_hours'])
        math_score = int(request.form['math_score'])
        history_score = int(request.form['history_score'])
        physics_score = int(request.form['physics_score'])
        chemistry_score = int(request.form['chemistry_score'])
        biology_score = int(request.form['biology_score'])
        english_score = int(request.form['english_score'])
        geography_score = int(request.form['geography_score'])
        total_score = float(request.form['total_score'])
        average_score = float(request.form['average_score'])

        hobbies_flag = True if hobbies else False

        recommendations = Recommendations(gender, part_time_job, hobbies_flag,
                                          weekly_self_study_hours, math_score, history_score,
                                          physics_score, chemistry_score, biology_score,
                                          english_score, geography_score, total_score, average_score)

        return render_template('results.html', recommendations=recommendations)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
