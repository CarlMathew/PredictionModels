import pandas as pd
from joblib import load
from fastapi import FastAPI, status, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model_routes import predictor


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["GET", "POST", "PUT", "DELETE"],
    allow_headers = ["Authorization", "Content-Type"]
)

app.include_router(predictor.router)









# model = load("learning_model/salary_predictor.joblib")
# test_data = pd.DataFrame([{
#     "Age": 23,
#     "Gender": "Female",
#     "Education Level": "Bachelor's",
#     "Job Title": "Programmer",
#     "Years of Experience": 0.5
# }])

# print(model.predict(test_data))


