import os
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # Allow local execution if pysqlite3 is not installed

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, session, redirect, url_for, flash
from supabase import create_client, Client
from functools import wraps
import numpy as np
import joblib
import pandas as pd
import logging
from flask import jsonify

from src.rag.chatbot import ThyroidBot

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Load model pipeline & label encoder ────────────────────
model = joblib.load('src/model/thyroid_model_v2.pkl')
label_encoder = joblib.load('src/model/label_encoder_v2.pkl')

THYROID_STAGE = {cls: cls for cls in label_encoder.classes_}

# Features the user submits via the form (8 features, same as original)
FORM_FEATURES = ['age', 'sex', 'on_thyroxine', 'on_antithyroid_meds', 'I131_treatment', 'TSH', 'T3', 'TT4']

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key_for_dev')

# ── Initialize Supabase ──
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
if supabase_url and supabase_key:
    supabase: Client = create_client(supabase_url, supabase_key)
else:
    logger.warning("Supabase credentials not found in environment variables.")
    supabase = None

@app.context_processor
def inject_user():
    return dict(user_email=session.get('user_email'))

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ── Initialize Chatbot ──
chatbot = ThyroidBot()
chatbot.initialize()

def build_full_feature_row(age, sex, on_thyroxine, on_antithyroid_meds,
                           I131_treatment, TSH, T3, TT4):
    """
    Build a complete feature DataFrame from the 8 user-submitted values.
    Engineered features and missing columns are computed or set to defaults.
    """
    # ── Base features ──
    row = {
        'age': float(age),
        'sex': sex,
        'on_thyroxine': int(on_thyroxine),
        'on_antithyroid_meds': int(on_antithyroid_meds),
        'I131_treatment': int(I131_treatment),
        'TSH': float(TSH),
        'T3': float(T3),
        'TT4': float(TT4),
    }

    # ── Columns not available from user form — set to defaults ──
    row['query_on_thyroxine'] = 0
    row['sick'] = 0
    row['pregnant'] = 0
    row['thyroid_surgery'] = 0
    row['query_hypothyroid'] = 0
    row['query_hyperthyroid'] = 0
    row['lithium'] = 0
    row['goitre'] = 0
    row['tumor'] = 0
    row['hypopituitary'] = 0
    row['psych'] = 0
    row['referral_source'] = 'other'
    row['T4U'] = np.nan  # Will be imputed by the pipeline
    row['FTI'] = np.nan  # Will be imputed by the pipeline

    # ── Engineered features ──
    row['TSH_T3_ratio'] = row['TSH'] / (row['T3'] + 1e-6)
    row['TSH_TT4_ratio'] = row['TSH'] / (row['TT4'] + 1e-6)
    row['T3_TT4_ratio'] = row['T3'] / (row['TT4'] + 1e-6)
    row['T4U_TT4_interaction'] = np.nan  # T4U is unknown

    # Age group
    age_val = row['age']
    if age_val <= 18:
        row['age_group'] = 'pediatric'
    elif age_val <= 45:
        row['age_group'] = 'young_adult'
    elif age_val <= 65:
        row['age_group'] = 'middle_aged'
    else:
        row['age_group'] = 'elderly'

    # Extreme flags
    row['extreme_TSH'] = int(row['TSH'] > 10 or row['TSH'] < 0.1)
    row['extreme_T3'] = int(row['T3'] > 4 or row['T3'] < 0.5)
    row['extreme_TT4'] = int(row['TT4'] > 200 or row['TT4'] < 40)

    # Missingness indicators (user provides TSH, T3, TT4 — so not missing)
    row['TSH_missing'] = 0
    row['T3_missing'] = 0
    row['TT4_missing'] = 0
    row['T4U_missing'] = 1  # Not provided
    row['FTI_missing'] = 1  # Not provided

    return pd.DataFrame([row])


@app.route('/health')
def health_check():
    return {"status": "healthy"}, 200


@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    csp = (
        "default-src 'self' https://overpass-api.de; "
        "script-src 'self' 'unsafe-inline' cdnjs.cloudflare.com cdn.jsdelivr.net unpkg.com; "
        "style-src 'self' 'unsafe-inline' fonts.googleapis.com unpkg.com; "
        "font-src 'self' fonts.gstatic.com; "
        "img-src 'self' data: cdn.builder.io https://*.tile.openstreetmap.org unpkg.com;"
    )
    response.headers['Content-Security-Policy'] = csp
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
        # Check for missing or empty fields
        if not all(field in request.form and str(request.form[field]).strip() != '' for field in FORM_FEATURES):
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

        # Build full feature row with engineered features
        features = build_full_feature_row(
            age, sex, on_thyroxine, on_antithyroid_meds,
            I131_treatment, TSH, T3, TT4
        )

        try:
            prediction_encoded = model.predict(features)[0]
            result = label_encoder.inverse_transform([prediction_encoded])[0]
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return render_template('index.html', error="Prediction failed due to an internal error."), 500

        return render_template('index.html', prediction=result, TSH=TSH, T3=T3, TT4=TT4)

    return render_template('index.html')

@app.route('/chat')
@login_required
def chat_page():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    user_message = data['message']
    
    try:
        bot_response = chatbot.ask(user_message)
        return jsonify({"response": bot_response})
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if supabase:
            try:
                response = supabase.auth.sign_up({"email": email, "password": password})
                if response.user:
                    return render_template('login.html', message="Signup successful! Please log in.")
                else:
                    return render_template('signup.html', error="Signup failed.")
            except Exception as e:
                return render_template('signup.html', error=str(e))
        else:
            return render_template('signup.html', error="Supabase not configured.")
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if supabase:
            try:
                response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                if response.user:
                    session['user_id'] = response.user.id
                    session['user_email'] = response.user.email
                    return redirect(url_for('home'))
                else:
                    return render_template('login.html', error="Invalid credentials.")
            except Exception as e:
                return render_template('login.html', error=str(e))
        else:
            return render_template('login.html', error="Supabase not configured.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    if supabase:
        supabase.auth.sign_out()
    return redirect(url_for('home'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', error="404 Error: The requested page was not found."), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Unhandled Exception: {e}", exc_info=True)
    return render_template('index.html', error="500 Error: An internal server error occurred."), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
