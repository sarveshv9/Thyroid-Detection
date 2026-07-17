from flask import Flask, render_template, request
import numpy as np
import joblib

model = joblib.load('src/model/RF1_model.pkl')

app = Flask(__name__)

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' cdnjs.cloudflare.com cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' fonts.googleapis.com; font-src 'self' fonts.gstatic.com;"
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/map')
def map_page():
    return render_template('map.html')

@app.route('/aboutus')
def about_us():
    return render_template('aboutus.html')

@app.route('/riskfac')
def risk_factors():
    return render_template('riskfac.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            age = float(request.form.get('age', 0))
            sex = request.form.get('sex', 'F')  
            on_thyroxine = int(request.form.get('on_thyroxine', 0))
            on_antithyroid_meds = int(request.form.get('on_antithyroid_meds', 0))
            I131_treatment = int(request.form.get('I131_treatment', 0))
            TSH = float(request.form.get('TSH', 0))
            T3 = float(request.form.get('T3', 0))
            TT4 = float(request.form.get('TT4', 0))
        except (ValueError, TypeError):
            return render_template('index.html', error="Invalid input. Please enter valid numerical values.")

        features = np.array([[age, 1 if sex == 'M' else 0, on_thyroxine, on_antithyroid_meds, I131_treatment, TSH, T3, TT4]])

        try:
            prediction = model.predict(features)[0]
        except Exception as e:
            return render_template('index.html', error="Prediction failed due to an internal error.")
        
        thyroid_stage = {
            0: "Normal",
            1: "Overt Hyperthyroidism",
            2: "Overt Hypothyroidism",
            3: "Subclinical Hyperthyroidism",
            4: "Subclinical Hypothyroidism",
            5: "Unclassified",
        }
        
        result = thyroid_stage.get(prediction, "Unknown")

        return render_template('index.html', prediction=result, TSH=TSH, T3=T3, TT4=TT4)


    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)
