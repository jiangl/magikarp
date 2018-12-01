from flask import Flask, request, jsonify
from core_insure.assessor.home_assessor import HomeAssessor
import random

def create_app():
    app = Flask(__name__.split('.')[0])
    assessor = HomeAssessor()

    @app.route('/')
    def test():
        moves = ["splash", "tackle", "flail"]
        return random.choice(moves)

    @app.route('/get_houses', methods=['POST'])
    def get_houses():
        # request.args (GET)
        # request.form.items()
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')

        fake_house = {
            'house_id': 33242,
            'risk_score': 19,
            'home_value': 134234,
            'insurance_quote': 200,
            'model_confidence': 95,
            'attributes': {},
            'latitude': latitude,
            'longitude': longitude
        }

        return jsonify([fake_house])

    return app

