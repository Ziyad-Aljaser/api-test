from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel

app = FastAPI()

# GET request
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

# GET request for item
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

# Load your models
model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define input data model
class InputFeatures(BaseModel):
    appearance: int
    assists: float
    minutes_played: int
    award: int

def preprocessing(input_features: InputFeatures):
    dict_f = {
        'appearance': input_features.appearance,
        'assists': input_features.assists,
        'minutes played': input_features.minutes_played,
        'award': input_features.award,
    }
    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    # Scale the input features
    scaled_features = scaler.transform([features_list])
    return scaled_features

@app.get("/predict")
def predict(input_features: InputFeatures):
    return preprocessing(input_features)


# Define the /predict endpoint to handle POST requests
@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}
