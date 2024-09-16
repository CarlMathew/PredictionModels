from pydantic import BaseModel
from typing import Optional, Dict


class SalaryPredictionParameters(BaseModel):
    Age: int
    Gender: str
    Education_Level: str
    JobTitle: str
    yearsOfExperience:float
