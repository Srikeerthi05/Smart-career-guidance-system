# pip install scikit-learn==1.3.2
# pip install numpy
# pip install flask


# load packages==============================================================
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the scaler, label encoder, model, and class names=====================
scaler = pickle.load(open("scaler.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))
class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']

# Profession details: image and description
profession_details = {
    'Lawyer': {
        'img': 'lawyer.jpg',
        'desc': 'Helps people with legal issues, represents clients in court.'
    },
    'Doctor': {
        'img': 'doctor.jpg',
        'desc': 'Treats illnesses, helps people stay healthy and live longer.'
    },
    'Government Officer': {
        'img': 'governmentofficer.jpg',
        'desc': 'Works for the public sector, manages departments or policies.'
    },
    'Artist': {
        'img': 'artist.jpg',
        'desc': 'Creates visual or performing art to express creativity.'
    },
    'Software Engineer': {
        'img': 'softwareengineer.jpg',
        'desc': 'Builds apps, websites, and systems using code.'
    },
    'Teacher': {
        'img': 'teacher.jpg',
        'desc': 'Educates students and shapes future generations.'
    },
    'Business Owner': {
        'img': 'businessowner.jpg',
        'desc': 'Runs and manages a business or startup.'
    },
    'Scientist': {
        'img': 'scientist.jpg',
        'desc': 'Explores and discovers new things through experiments.'
    },
    'Banker': {
        'img': 'banker.jpg',
        'desc': 'Manages money, loans, and financial services.'
    },
    'Writer': {
        'img': 'writer.jpg',
        'desc': 'Creates stories, articles, and content that inspire.'
    },
    'Accountant': {
        'img': 'accountant.jpg',
        'desc': 'Handles finances, taxes, and keeps records accurate.'
    },
    'Designer': {
        'img': 'designer.jpg',
        'desc': 'Creates visuals and layouts for products or media.'
    },
    'Construction Engineer': {
        'img': 'construct engineer.jpg',
        'desc': 'Plans and supervises building structures and infrastructure.'
    },
    'Game Developer': {
        'img': 'gamedeveloper.jpg',
        'desc': 'Designs and codes exciting video games.'
    },
    'Stock Investor': {
        'img': 'stockinvestor.jpg',
        'desc': 'Buys and sells stocks to grow wealth over time.'
    },
    'Real Estate Developer': {
        'img': 'realestatedeveloper.jpg',
        'desc': 'Builds and sells properties for living or business.'
    }
}



 
def Recommendations(gender, part_time_job,hobbies,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    hobbies_encoded = 1 if hobbies else 0  
     
    absence_days = 0
    extracurricular_activities = 0
    # Create feature array
    feature_array = np.array([gender_encoded, part_time_job_encoded,hobbies_encoded, 
                               weekly_self_study_hours, math_score, history_score, physics_score,
                               chemistry_score, biology_score, english_score, geography_score, total_score,
                               average_score])

    # Scale features
    scaled_features = scaler.transform(feature_array.reshape(1,-1))

    # Predict using the model
    probabilities = model.predict_proba(scaled_features)

    # Get top five predicted classes along with their probabilities
    top_classes_idx = np.argsort(-probabilities[0])[:3]
    top_classes_names_probs = []
    for idx in top_classes_idx:
        name = class_names[idx]
        prob = probabilities[0][idx]
        detail = profession_details.get(name, {"img": "default.png", "desc": "No description available."})
    
        top_classes_names_probs.append({
            "name": name,
            "prob": prob,
            "img": detail["img"],
            "desc": detail["desc"]
        })
        
    return top_classes_names_probs



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

@app.route('/pred', methods=['POST','GET'])
@app.route('/pred', methods=['POST', 'GET'])
def pred():
    if request.method == 'POST':
        try:
            gender = request.form['gender']
            part_time_job = request.form['part_time_job'].lower() == 'true'
            hobbies = request.form.getlist('hobbies')
            hobbies_flag = bool(hobbies)

            # Convert scores safely
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

            # Call the recommendation function
            recommendations = Recommendations(
                gender, part_time_job, hobbies_flag,
                weekly_self_study_hours, math_score, history_score, physics_score,
                chemistry_score, biology_score, english_score, geography_score,
                total_score, average_score
            )

            return render_template('results.html', recommendations=recommendations)
        
        except Exception as e:
            print("Error:", e)  # This will help you debug
            return render_template('error.html', message=str(e))  # optional error page
        
    return render_template('home.html')



if __name__ == '__main__':
    app.run(debug=True)

