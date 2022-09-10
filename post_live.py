import requests
import json

data = {
  "age": 22,
  "workclass": "State-gov",
  "fnlgt": 59951,
  "education": "Bachelors",
  "education_num": 4,
  "marital_status": "Married-civ-spouse",
  "occupation": "Adm-clerical",
  "relationship": "Wife",
  "race": "White",
  "sex": "Female",
  "capital_gain": 2174,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "Puerto-Rico"
}

response = requests.post('https://mlprod.herokuapp.com/predict/', data=json.dumps(data))

print(response.status_code)
print(response.json())
