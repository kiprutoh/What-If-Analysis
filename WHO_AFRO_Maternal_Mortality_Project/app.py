import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

model = joblib.load("maternal_mortality_model.pkl")

st.set_page_config(
    page_title="WHO AFRO Maternal Mortality What-if Analysis",
    layout="wide"
)

st.title("WHO AFRO Maternal Mortality What-if Analysis")

st.sidebar.header("Scenario Inputs")

sba = st.sidebar.slider(
    "Skilled Birth Attendance (%)",
    0,
    100,
    70
)

fertility = st.sidebar.slider(
    "Fertility Rate",
    1.0,
    8.0,
    4.5
)

literacy = st.sidebar.slider(
    "Female Literacy (%)",
    0,
    100,
    65
)

rural = st.sidebar.slider(
    "Rural Population (%)",
    0,
    100,
    55
)

scenario = pd.DataFrame({
    "Skilled Birth Attendance": [sba],
    "Fertility Rate": [fertility],
    "Female Literacy": [literacy],
    "Rural Population": [rural]
})

prediction = model.predict(scenario)[0]

st.metric(
    label="Predicted Maternal Mortality Ratio",
    value=round(prediction, 1)
)

fig = px.bar(
    x=["Predicted MMR"],
    y=[prediction],
    labels={"x": "Scenario", "y": "MMR"},
    title="Predicted Maternal Mortality Ratio"
)

st.plotly_chart(fig, use_container_width=True)

if st.button("Download Scenario"):

    scenario["Predicted_MMR"] = prediction

    scenario.to_excel("scenario_output.xlsx", index=False)

    with open("scenario_output.xlsx", "rb") as file:

        st.download_button(
            label="Download Excel File",
            data=file,
            file_name="scenario_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )