import pandas as pd
import joblib

model = joblib.load("maternal_mortality_model.pkl")

def run_scenario(
    skilled_birth_attendance,
    fertility_rate,
    female_literacy,
    rural_population
):

    scenario = pd.DataFrame({
        "Skilled Birth Attendance": [skilled_birth_attendance],
        "Fertility Rate": [fertility_rate],
        "Female Literacy": [female_literacy],
        "Rural Population": [rural_population]
    })

    prediction = model.predict(scenario)[0]

    return prediction

predicted_mmr = run_scenario(
    skilled_birth_attendance=85,
    fertility_rate=4.0,
    female_literacy=70,
    rural_population=60
)

print(predicted_mmr)