import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware


# Load the saved preprocessing and prediction pipeline.
model = joblib.load("Mental_Health_Model.pkl")

# Country groups used by the notebook and prediction pipeline.
top_countries = [
    "Other",
    "India",
    "USA",
    "Canada",
    "Australia",
    "UK",
    "Germany",
    "Mexico",
    "Turkey",
    "France",
]

app = FastAPI(
    title="Mental Health Score Prediction API",
    description=(
        "API for an educational machine learning project that estimates a "
        "reported mental health score from social media usage and daily habits. "
        "This is not a medical assessment or diagnosis."
    ),
    version="1.0.0",
)

# Allow the web frontend to send requests to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Validate the information submitted by the user.
class StudentData(BaseModel):
    age: int = Field(..., ge=10, le=100)
    gender: Literal["Male", "Female"]
    country: str
    academic_level: Literal["Undergraduate", "Graduate", "High School"]
    most_used_platform: Literal[
        "Facebook",
        "LinkedIn",
        "Instagram",
        "Snapchat",
        "Twitter",
        "YouTube",
        "TikTok",
        "LINE",
        "KakaoTalk",
        "VKontakte",
        "WhatsApp",
        "WeChat",
    ]
    purpose_of_use: Literal["Networking", "Education", "Entertainment", "News"]
    avg_daily_usage_hours: float = Field(..., ge=0, le=24)
    daily_unlocks: int = Field(..., ge=0)
    study_hours: float = Field(..., ge=0, le=24)
    physical_activity_hours: float = Field(..., ge=0, le=24)
    sleep_hours_per_night: float = Field(..., ge=0, le=24)
    stress_level: Literal["Medium", "Low", "Very High", "High"]


# Define the format of the prediction returned by the API.
class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


@app.get("/")
def greet():
    return {
        "message": "Mental Health Score Prediction API",
        "status": "running",
        "documentation": "/docs",
        "note": "Educational prediction only; not a medical diagnosis.",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentData):
    # Keep the country group consistent with the notebook's feature format.
    country_group = data.country if data.country in top_countries else "Other"

    # Arrange the submitted values as one row for the saved pipeline.
    input_row = pd.DataFrame(
        [
            {
                "Age": data.age,
                "Gender": data.gender,
                "Country": data.country,
                "Academic_Level": data.academic_level,
                "Most_Used_Platform": data.most_used_platform,
                "Purpose_Of_Use": data.purpose_of_use,
                "Avg_Daily_Usage_Hours": data.avg_daily_usage_hours,
                "Daily_Unlocks": data.daily_unlocks,
                "Study_Hours": data.study_hours,
                "Physical_Activity_Hours": data.physical_activity_hours,
                "Sleep_Hours_Per_Night": data.sleep_hours_per_night,
                "Stress_Level": data.stress_level,
                "Grouped_country": country_group,
            }
        ]
    )

    # Return the predicted score rounded to two decimal places.
    prediction = model.predict(input_row)[0]
    return PredictionResponse(
        predicted_mental_health_score=round(float(prediction), 2)
    )
