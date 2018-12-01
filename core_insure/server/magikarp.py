from flask import Flask, request, jsonify
from core_insure.assessor.home_assessor import HomeAssessor
from core_insure.data.dataloader import DataLoader
import random
from ruamel.yaml import YAML
from celery import Celery


def create_app(config_file):
    yaml = YAML()
    config = yaml.load(config_file)

    app = Flask(__name__.split('.')[0])
    celery = Celery(app.name, broker=config.get('broker_url'))
    celery.config_from_object('celeryconfig')

    assessor = HomeAssessor(config.get('assessor'))
    dataloader = DataLoader(config.get('data'))

    def run_save_claim_prediction(house_id):
        attributes = dataloader.load_attributes(house_id)
        claim = assessor.predict_from_attributes(attributes)
        dataloader.update_claim(house_id, claim)

    @celery.task(bind=True)
    def update_claim_predictions(house_ids=None):
        if house_ids:
            for house_id in house_ids:
                run_save_claim_prediction(house_id)
        else:
            pass # db generator, run for all

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
        house_id = json_params.get('house_id')
        attributes = json_params.get('attributes')
        dataloader.save_attributes(house_id, attributes)
        all_attributes = dataloader.load_attributes(house_id)
        claim_pred = assessor.predict_from_attributes(all_attributes)
        return claim_pred

    @app.route('/update_model', methods=['POST'])
    def update_model():
        json_params = request.get_json()
        house_id = json_params.get('house_id')
        claim_amount = json_params.get('claim_amount')
        dataloader.update_claim(house_id, claim_amount)
        all_attributes = dataloader.load_attributes(house_id)
        assessor.train((all_attributes, claim_amount))
        update_claim_predictions.delay()

    return app

