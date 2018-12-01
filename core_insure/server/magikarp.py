from flask import Flask, request, jsonify
from core_insure.assessor.home_assessor import HomeAssessor
import random
from ruamel.yaml import YAML


def create_app(config_file):
    app = Flask(__name__.split('.')[0])
    yaml = YAML()
    config = yaml.load(config_file)
    assessor = HomeAssessor(config.get('assessor'))

    @app.route('/')
    def test():
        moves = ["splash", "tackle", "flail"]
        return random.choice(moves)

    @app.route('/get_houses', methods=['POST'])
    def get_houses():
        # request.args (GET)
        # request.form.items()
        latitude = float(request.form.get('latitude'))
        longitude = float(request.form.get('longitude'))

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

