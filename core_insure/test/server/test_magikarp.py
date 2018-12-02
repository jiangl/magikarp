import pytest
from core_insure.server.magikarp import create_app
from core_insure.assessor.home_assessor import Attributes
import json

@pytest.fixture
def client():
    config_file = open('./config.yaml', 'r')
    app = create_app(config_file)
    return app.test_client()

@pytest.fixture
def fake_house():
    return {
        'house_id': 33242,
        'risk_score': 19,
        'home_value': 134234,
        'insurance_quote': 200,
        'model_confidence': 95,
        'attributes': {}
    }

def convert_string(value):
    return value.decode('utf-8')

def test_basic(client):
    output = client.get('/')
    status = output.status_code
    value = convert_string(output.data)
    assert status == 200
    assert value in ["flail", "splash", "tackle"]

def test_get_houses(client, fake_house):
    extra_data = dict(
        latitude=7,
        longitude=20
    )
    fake_house.update(extra_data)

    output = client.post('/get_houses', json=extra_data)
    status = output.status_code
    values = json.loads(output.data)
    expected_output = [fake_house]

    assert status == 200
    assert values == expected_output

def test_update_attributes(client):
    fake_features = {
        'ZIPCODE': 11226,
        'FLOOD_DAMAGE': 0,
        'ROOF_DAMAGE': 1,
        'INCOME': 34233
    }

    output = client.post('/update_attribute', json=dict(house_id=0,
                                                        attributes=fake_features))
    status = output.status_code
    values = output.data
    assert status == 200