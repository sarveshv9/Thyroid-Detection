from flask import Flask, render_template, request
import numpy as np
import joblib

model = joblib.load('src/model/RF1_model.pkl')

app = Flask(__name__)


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
       
        age = float(request.form['age'])
        sex = request.form['sex']  
        on_thyroxine = int(request.form['on_thyroxine'])
        on_antithyroid_meds = int(request.form['on_antithyroid_meds'])
        I131_treatment = int(request.form['I131_treatment'])
        TSH = float(request.form['TSH'])
        T3 = float(request.form['T3'])
        TT4 = float(request.form['TT4'])

        features = np.array([[age, 1 if sex == 'M' else 0, on_thyroxine, on_antithyroid_meds, I131_treatment, TSH, T3, TT4]])

        prediction = model.predict(features)[0]
        
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
    app.run(debug=True)
