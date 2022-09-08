from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_api_locally_get_root():
    r = client.get('/')
    assert r.status_code == 200


def test_get_malformed():
    r = client.get("/items")
    assert r.status_code != 200


def test_predict_high():
    request = client.post("/predict", json={'workclass': 'Private',
                                            'education': 'HS-grad',
                                            'marital_status': 'Never-married',
                                            'occupation': 'Prof-specialty',
                                            'relationship': 'Not-in-family',
                                            'race': 'White',
                                            'sex': 'Male',
                                            'native_country': 'United-States',
                                            'age': 33,
                                            'education_num': 5,
                                            'capital_gain': 2174,
                                            'capital_loss': 0,
                                            'fnlgt': 149184,
                                            'hours_per_week': 40,
                                            })
    assert request.status_code == 200
    assert request.json() == {'prediction': ">50k"}


def test_predict_low():
    request = client.post("/predict", json={'workclass': 'Private',
                                            'education': 'HS-grad',
                                            'marital_status': 'Never-married',
                                            'occupation': 'Prof-specialty',
                                            'relationship': 'Not-in-family',
                                            'race': 'White',
                                            'sex': 'Male',
                                            'native_country': 'United-States',
                                            'age': 25,
                                            'education_num': 2,
                                            'capital_gain': 2174,
                                            'capital_loss': 0,
                                            'fnlgt': 73184,
                                            'hours_per_week': 30,
                                            })
    assert request.status_code == 200
    assert request.json() == {'prediction': ">50k"}


def test_home():
    request = client.get("/")
    print(request.content)
    assert request.status_code == 200
    assert request.json() == {"home": "This is the home; to run inference, insert URL 'http://127.0.0.1:8000/predict"}
