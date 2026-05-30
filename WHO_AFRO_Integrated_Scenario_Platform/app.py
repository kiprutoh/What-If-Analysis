
import streamlit as st
import pandas as pd

st.set_page_config(page_title="WHO AFRO Health Systems Scenario Platform", layout="wide")

st.title("WHO AFRO Health Systems Scenario Platform")

analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Maternal Mortality", "Universal Health Coverage (UHC)"]
)

if analysis_type == "Maternal Mortality":
    st.header("Maternal Mortality What-if Analysis")

    sba = st.sidebar.slider("Skilled Birth Attendance (%)", 0, 100, 70)
    anc = st.sidebar.slider("ANC4 Coverage (%)", 0, 100, 65)
    literacy = st.sidebar.slider("Female Literacy (%)", 0, 100, 60)
    fertility = st.sidebar.slider("Fertility Rate", 1.0, 8.0, 4.0)

    mmr = 800 - (4*sba) - (2*literacy) + (50*fertility) - (1.5*anc)

    st.metric("Predicted Maternal Mortality Ratio", round(mmr,1))

else:
    st.header("Universal Health Coverage (UHC) What-if Analysis")

    coverage = st.slider("Service Coverage Score",0,100,60)
    workforce = st.slider("Workforce Score",0,100,50)
    financing = st.slider("Financial Protection Score",0,100,40)
    infrastructure = st.slider("Infrastructure Score",0,100,50)
    medicines = st.slider("Medicines Availability Score",0,100,60)
    his = st.slider("HIS & Digital Health Score",0,100,55)
    governance = st.slider("Governance Score",0,100,50)
    population = st.slider("Population & Equity Score",0,100,45)

    uhc = (
        0.30*coverage +
        0.15*workforce +
        0.15*financing +
        0.10*infrastructure +
        0.10*medicines +
        0.10*his +
        0.05*governance +
        0.05*population
    )

    st.metric("UHC Composite Score", round(uhc,1))
