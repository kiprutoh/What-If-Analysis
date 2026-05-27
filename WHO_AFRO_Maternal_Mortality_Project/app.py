import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

st.set_page_config(
    page_title="WHO AFRO Maternal Mortality What-if Analysis",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = Path("models/maternal_mortality_model.pkl")
    return joblib.load(model_path)

model = load_model()

st.title("WHO AFRO Maternal Mortality What-if Analysis")

st.markdown(
    "Interactive scenario simulation platform for maternal mortality analysis across WHO AFRO countries."
)

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

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Predicted Maternal Mortality Ratio",
        value=round(prediction, 1)
    )

with col2:
    baseline = 450
    reduction = baseline - prediction
    st.metric(
        label="Estimated Reduction from Baseline",
        value=round(reduction, 1)
    )

fig = px.bar(
    x=["Predicted MMR"],
    y=[prediction],
    labels={"x": "Scenario", "y": "MMR"},
    title="Predicted Maternal Mortality Ratio"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Scenario Data")

scenario["Predicted_MMR"] = round(prediction, 2)

st.dataframe(scenario, use_container_width=True)

output_file = "scenario_output.xlsx"
scenario.to_excel(output_file, index=False)

with open(output_file, "rb") as file:
    st.download_button(
        label="Download Scenario Excel File",
        data=file,
        file_name="scenario_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )