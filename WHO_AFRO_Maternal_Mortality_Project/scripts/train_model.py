import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_excel("WHO_AFRO_Processed_Data.xlsx")
df = df.dropna()

X = df[
    [
        "Skilled Birth Attendance",
        "Fertility Rate",
        "Female Literacy",
        "Rural Population"
    ]
]

y = df["Maternal Mortality Ratio"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)

print(f"MAE: {mae}")

joblib.dump(model, "maternal_mortality_model.pkl")

print("Model saved")