import unittest
import json
import sys
import os

# Add the project root to the sys path so we can import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, EXPECTED_FEATURES

class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        # Create a test client
        self.app = app.test_client()
        self.app.testing = True

    # 1. Routing / Unit Tests
    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_map_page(self):
        response = self.app.get('/map')
        self.assertEqual(response.status_code, 200)

    def test_about_us_page(self):
        response = self.app.get('/aboutus')
        self.assertEqual(response.status_code, 200)

    def test_risk_factors_page(self):
        response = self.app.get('/riskfac')
        self.assertEqual(response.status_code, 200)

    def test_404_error_handler(self):
        response = self.app.get('/nonexistent_route')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b"404 Error", response.data)

    # 2. Integration / API Tests (Invalid Inputs)
    def test_predict_get_request(self):
        # GET should return the HTML form
        response = self.app.get('/predict')
        self.assertEqual(response.status_code, 200)

    def test_predict_post_missing_fields(self):
        response = self.app.post('/predict', data={
            'age': '30',
            'sex': '0',
            # missing TSH, T3, etc.
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Missing required input fields", response.data)

    def test_predict_post_empty_fields(self):
        response = self.app.post('/predict', data={
            'age': '30',
            'sex': '0',
            'on_thyroxine': '0',
            'on_antithyroid_meds': '0',
            'I131_treatment': '0',
            'TSH': '',
            'T3': '',
            'TT4': ''
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Missing required input fields", response.data)

    def test_predict_post_invalid_types(self):
        response = self.app.post('/predict', data={
            'age': 'Thirty',  # String instead of float
            'sex': '0',
            'on_thyroxine': '0',
            'on_antithyroid_meds': '0',
            'I131_treatment': '0',
            'TSH': '0.5',
            'T3': '1.5',
            'TT4': '100'
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Invalid input. Please enter valid numerical values.", response.data)

    # 3. ML Validation / Regression Tests (Valid Inputs)
    def test_predict_post_valid_normal_thyroid(self):
        # Based on notebook rules: TSH=2.0, T3=2.0, TT4=100.0 is Normal
        response = self.app.post('/predict', data={
            'age': '45',
            'sex': '0',
            'on_thyroxine': '0',
            'on_antithyroid_meds': '0',
            'I131_treatment': '0',
            'TSH': '2.0',
            'T3': '2.0',
            'TT4': '100.0'
        })
        self.assertEqual(response.status_code, 200)
        # Check if the HTML returns the prediction
        self.assertIn(b"Normal", response.data)

    def test_predict_post_valid_hyperthyroid(self):
        # TSH < 0.35 and (T3 > 3.0 or TT4 > 180) -> Overt Hyperthyroidism
        response = self.app.post('/predict', data={
            'age': '45',
            'sex': '1',
            'on_thyroxine': '0',
            'on_antithyroid_meds': '0',
            'I131_treatment': '0',
            'TSH': '0.1',
            'T3': '4.5',
            'TT4': '200.0'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Overt Hyperthyroidism", response.data)

if __name__ == '__main__':
    unittest.main()
