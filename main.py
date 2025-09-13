from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
with open('crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Simulated soil data by location
def get_soil_defaults(location: str):
    soil_data = {
        "Bheemunipatnam": {"N": 80, "P": 40, "K": 35, "ph": 6.8, "rainfall": 220},
        "Nagpur": {"N": 60, "P": 30, "K": 25, "ph": 7.2, "rainfall": 180},
        "Coimbatore": {"N": 75, "P": 38, "K": 32, "ph": 6.5, "rainfall": 240},
        "Lucknow": {"N": 65, "P": 34, "K": 28, "ph": 7.1, "rainfall": 190},
    }
    return soil_data.get(location, {"N": 70, "P": 35, "K": 30, "ph": 7.0, "rainfall": 200})

# Simulated weather data
def get_weather(location: str):
    weather_data = {
        "Bheemunipatnam": {"temperature": 27.5, "humidity": 78},
        "Nagpur": {"temperature": 30.0, "humidity": 65},
        "Coimbatore": {"temperature": 26.0, "humidity": 80},
        "Lucknow": {"temperature": 29.0, "humidity": 70},
    }
    return weather_data.get(location, {"temperature": 28.0, "humidity": 75})

# Simulated location detection
def get_location_name(lat, lon):
    # For now, simulate based on known coordinates
    return "Bheemunipatnam"  # Later, replace with real reverse geocoding API

# Input model for POST request
class GeoInput(BaseModel):
    latitude: float
    longitude: float

# Crop recommendation endpoint
@app.post("/geo-recommend")
def recommend_crop_geo(data: GeoInput):
    location = get_location_name(data.latitude, data.longitude)
    soil = get_soil_defaults(location)
    weather = get_weather(location)

    input_data = [[
        soil["N"], soil["P"], soil["K"],
        weather["temperature"], weather["humidity"],
        soil["ph"], soil["rainfall"]
    ]]

    prediction = model.predict(input_data)[0]
    return {"location": location, "recommended_crop": prediction}
