from flask import Flask, request, jsonify
from assessor.home_assessor import HomeAssessor
from dataio.dataloader import DataLoader
import random
from ruamel.yaml import YAML
from celery import Celery


def create_app(config_file):
    yaml = YAML()
    config = yaml.load(config_file)

    app = Flask(__name__.split('.')[0])
    celery = Celery(app.name, broker=config.get('celery').get('broker_url'))
    celery.config_from_object('celeryconfig')

    assessor = HomeAssessor(config.get('assessor'))
    assessor.load()
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
            # TODO: db generator, run for all
            pass

    @app.route('/')
    def test():
        moves = ["splash", "tackle", "flail"]
        return random.choice(moves)

    @app.route('/get_houses', methods=['POST'])
    def get_houses():
        json_params = request.get_json()
        lat_long1 = json_params.get('lat_long1', 0)
        lat_long2 = json_params.get('lat_long2', 0)
        house_list = dataloader.load_houses(lat_long1, lat_long2)
        return jsonify(house_list)

    @app.route('/update_attribute', methods=['POST'])
    def update_attribute():
        json_params = request.get_json()
        house_id = json_params.get('house_id')
        attributes = json_params.get('attributes')
        dataloader.save_attributes(house_id, attributes)
        all_attributes = dataloader.load_attributes(house_id)
        claim_pred = assessor.predict_from_attributes(all_attributes)
        return jsonify({'claim_pred': float(claim_pred)})

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

