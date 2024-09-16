from fastapi import FastAPI, status, HTTPException, Depends, APIRouter
from schemas import *
import pandas as pd
from joblib import load

salaryPredictorModel = load("learning_model/salary_predictor.joblib")

router = APIRouter(tags=["Predictors"])

@router.get("/Predict")
async def HelloWorld():
    return {"Status":"Success"}

@router.get("/PredictSalary", status_code=status.HTTP_200_OK)
async def PredictSalary(payload: SalaryPredictionParameters = Depends(SalaryPredictionParameters)):
    test_data = pd.DataFrame([{
        "Age": payload.Age,
        "Gender": payload.Gender,
        "Education Level": payload.Education_Level,
        "Job Title": payload.JobTitle,
        "Years of Experience": payload.yearsOfExperience
    }])
    prediction = salaryPredictorModel.predict(test_data)
    return {"Prediction Salary": prediction[0]}
    



