from flask import Flask, request
app = Flask(__name__)

@app.route('/get_houses', methods=['POST'])
def get_houses():
    latitude = request.args.get('latitude', 0)
    longitude = request.args.get('latitude', 0)

    fake_house = {
        'house_id': 33242,
        'risk_score': 19,
        'home_value': 134234,
        'insurance_quote': 200,
        'model_confidence': 95,
        'attributes': {
            ''
        }
    }
    return [fake_house]
