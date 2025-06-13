from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('recommend.html')

@app.route('/pred', methods=['POST'])
def Recommendations():
    if request.method == 'POST':
        gender = request.form['gender']
        part_time_job = request.form['part_time_job']
        hobbies = request.form['extracurricular_activities']  # still using this name from the HTML
        intermediate_score = float(request.form['intermediate_score'])

        # Subject Scores
        history_score = float(request.form['history_score'])
        physics_score = float(request.form['physics_score'])
        chemistry_score = float(request.form['chemistry_score'])
        biology_score = float(request.form['biology_score'])
        english_score = float(request.form['english_score'])
        geography_score = float(request.form['geography_score'])
        total_score = float(request.form['total_score'])
        average_score = float(request.form['average_score'])

        # Encode categorical data
        gender_encoded = 1 if gender.lower() == 'male' else 0
        part_time_job_encoded = 1 if part_time_job.lower() == 'true' else 0
        hobbies_encoded = 1 if hobbies.lower() == 'true' else 0

        # Feature Array (adjust if you retrain your model)
        feature_array = np.array([[gender_encoded, part_time_job_encoded, hobbies_encoded,
                                   intermediate_score, history_score, physics_score,
                                   chemistry_score, biology_score, english_score,
                                   geography_score, total_score, average_score]])

        prediction = model.predict(feature_array)
        output = prediction[0]

        return render_template('recommend.html', prediction_text=f'Recommended Career Path: {output}')

if __name__ == '__main__':
    app.run(debug=True)