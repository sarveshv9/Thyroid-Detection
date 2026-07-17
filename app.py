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
        required_fields = ['age', 'sex', 'on_thyroxine', 'on_antithyroid_meds', 'I131_treatment', 'TSH', 'T3', 'TT4']
        
        # Check for missing or empty fields
        if not all(field in request.form and str(request.form[field]).strip() != '' for field in required_fields):
            return render_template('index.html', error="Missing required input fields. Please fill out the entire form."), 400

        try:
            age = float(request.form.get('age'))
            sex = request.form.get('sex')  
            on_thyroxine = int(request.form.get('on_thyroxine'))
            on_antithyroid_meds = int(request.form.get('on_antithyroid_meds'))
            I131_treatment = int(request.form.get('I131_treatment'))
            TSH = float(request.form.get('TSH'))
            T3 = float(request.form.get('T3'))
            TT4 = float(request.form.get('TT4'))
        except (ValueError, TypeError):
            return render_template('index.html', error="Invalid input. Please enter valid numerical values."), 400

        import pandas as pd
        feature_names = ['age', 'sex', 'on_thyroxine', 'on_antithyroid_meds', 'I131_treatment', 'TSH', 'T3', 'TT4']
        features = pd.DataFrame([[age, int(sex), on_thyroxine, on_antithyroid_meds, I131_treatment, TSH, T3, TT4]], columns=feature_names)

        try:
            prediction = model.predict(features)[0]
        except Exception as e:
            return render_template('index.html', error="Prediction failed due to an internal error."), 500

        
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

@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', error="404 Error: The requested page was not found."), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('index.html', error="500 Error: An internal server error occurred."), 500

if __name__ == '__main__':
    app.run(debug=False)
