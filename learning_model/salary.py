

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import dump

df = pd.read_csv("Salary Data.csv")
df.dropna(inplace=True)


X = df.drop("Salary", axis = 1)
y = df["Salary"]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 32)

one_hot = OneHotEncoder(sparse_output=False)
ordinal = OrdinalEncoder()
scaler = StandardScaler()
vectorizer = TfidfVectorizer()

print(df.select_dtypes(include = "object").nunique())
one_hot_column = ["Education Level"]
ordinalColumn = ["Gender"]
scalerColumn = ['Age', 'Years of Experience']
pipe_one_hot = Pipeline([("one_hot", one_hot)])
pipe_ordinal = Pipeline([("ordinal", ordinal)])
pipe_scaler = Pipeline([("scaler", scaler)])


preprocessor = ColumnTransformer([
    ("one_hot", pipe_one_hot, one_hot_column),
    ("ordinal", pipe_ordinal, ordinalColumn),
    ("scaler", pipe_scaler, scalerColumn),
    ('vectorizer', vectorizer, "Job Title")
    ], remainder = "passthrough")

model = LinearRegression()


final_pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

final_pipeline.fit(X_train, y_train)
y_pred = final_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)


dump(final_pipeline, "salary_predictor.joblib")
