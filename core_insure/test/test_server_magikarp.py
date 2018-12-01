import pytest
from core_insure.server.magikarp import create_app
import json

@pytest.fixture
def client():
    app = create_app()
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
        latitude=17,
        longitude=20
    )
    output = client.post('/get_houses', data=extra_data)
    status = output.status_code
    values = json.loads(output.data)
    expected_output = [fake_house.update(extra_data)]

    assert status == 200
    assert values == expected_output