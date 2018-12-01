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
        json_params = request.get_json()
        latitude = json_params.get('latitude', 0)
        longitude = json_params.get('longitude', 0)
        # pull from db
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

    @app.route('/update_attribute', methods=['POST'])
    def update_attribute():
        json_params = request.get_json()
        attributes = json_params.get('attributes')
        # update in db
        # pull full db record
        all_records = attributes
        claim_pred = assessor.predict_from_attributes(all_records)
        return claim_pred



    return app

