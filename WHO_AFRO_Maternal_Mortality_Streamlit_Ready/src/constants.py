"""WHO African Region countries and World Bank indicator codes."""

# WHO African Region member states (ISO3)
AFRO_ISO3 = [
    "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD",
    "COM", "COG", "COD", "CIV", "GNQ", "ERI", "SWZ", "ETH", "GAB", "GMB",
    "GHA", "GIN", "GNB", "KEN", "LSO", "LBR", "MDG", "MWI", "MLI", "MRT",
    "MUS", "MOZ", "NAM", "NER", "NGA", "RWA", "STP", "SEN", "SYC", "SLE",
    "ZAF", "SSD", "SDN", "TGO", "UGA", "TZA", "ZMB", "ZWE",
]

# World Bank API indicator codes (public, documented at data.worldbank.org)
INDICATORS = {
    "mmr": "SH.STA.MMRT",           # Maternal mortality ratio (per 100k live births)
    "sba": "SH.STA.BRTW.ZS",        # Births attended by skilled health staff (%)
    "fertility": "SP.DYN.TFRT.IN",  # Fertility rate (births per woman)
    "ado_fertility": "SP.ADO.TFRT", # Adolescent fertility rate (births per 1,000 women 15-19)
    "contraception_modern": "SP.DYN.CONM.ZS",  # Contraceptive prevalence, modern (% married women 15-49)
    "anc4": "SP.DYN.4ANTE.ZS",      # Antenatal care coverage, at least 4 visits (%)
    "poverty_215": "SI.POV.DDAY",   # Poverty headcount ratio at $2.15/day (2017 PPP) (% of population)
    "gov_health_exp": "SH.XPD.GHED.GD.ZS",  # Domestic general government health expenditure (% of GDP)
    "hiv_prev": "SH.DYN.AIDS.ZS",   # Prevalence of HIV, total (% of population ages 15-49)
    "malaria_inc": "SH.MLR.INCD.P3",# Incidence of malaria (per 1,000 population at risk)
    "anaemia_wra": "SH.ANM.ALLW.ZS",# Anaemia prevalence in women of reproductive age (% ages 15-49)
    "literacy": "SE.ADT.LITR.FE.ZS",  # Literacy rate, adult female (%)
    "literacy_total": "SE.ADT.LITR.ZS",  # Literacy rate, adult total (%)
    "rural": "SP.RUR.TOTL.ZS",      # Rural population (% of total)
}

FEATURE_COLUMNS = [
    # Health system variables
    "Skilled Birth Attendance",
    "Emergency Obstetric Care Coverage",
    "Caesarean Section Rate",
    "Health Workforce Density",
    "Midwife Density",
    "Facility Delivery Rate",
    "Blood Availability Index",
    "Referral Efficiency Index",
    # Reproductive variables
    "Contraception (modern)",
    "Adolescent Fertility Rate",
    "ANC4 Coverage",
    # Disease variables
    "HIV Prevalence (15-49)",
    "Malaria Incidence",
    "Anaemia (women 15-49)",
    # Socioeconomic variables
    "Female Literacy",
    "Poverty ($2.15/day, %)",
    "Rural Population",
    "Fertility Rate",
    "Adult Literacy",
    # Infrastructure variables
    "Travel Time to Facility (min)",
    "Road Access Index",
    "Ambulance Coverage Index",
    "Health Facility Density (per 10k)",
    # Governance / data variables
    "Gov Health Expenditure (% GDP)",
    "Maternal Death Reviews Coverage",
    "Reporting Completeness",
    "Reporting Timeliness",
    "HIS Maturity Index",
    "Civil Registration Completeness",
    # Emergency & shock variables
    "Epidemic Occurrence (0/1)",
    "Conflict Intensity Index",
    "Flood Occurrence (0/1)",
    "Supply Chain Disruption Index",
    "Workforce Strike Duration (days)",
]

# Directionality mapping used to enforce "good vs bad" interpretation.
#
# We convert all features to a unified "risk-oriented" space for modeling:
#   higher (risk-oriented value)  => higher expected MMR (worse)
#
# For variables where "higher is better" (protective), the risk-oriented value
# is the negative of the raw value (multiplier = -1).
#
# For variables where "lower is better" (risk factor), we keep as-is
# (multiplier = +1).
#
# This enables sign checks / monotonic expectations in the regression.
FEATURE_TO_RISK_MULTIPLIER: dict[str, float] = {
    # Health system: higher is better (protective)
    "Skilled Birth Attendance": -1.0,
    "Emergency Obstetric Care Coverage": -1.0,
    "Caesarean Section Rate": -1.0,
    "Health Workforce Density": -1.0,
    "Midwife Density": -1.0,
    "Facility Delivery Rate": -1.0,
    "Blood Availability Index": -1.0,
    "Referral Efficiency Index": -1.0,
    # Reproductive
    "Contraception (modern)": -1.0,
    "ANC4 Coverage": -1.0,
    "Adolescent Fertility Rate": +1.0,  # lower is better
    # Socioeconomic / demographic
    "Fertility Rate": +1.0,
    "Female Literacy": -1.0,
    "Adult Literacy": -1.0,
    "Poverty ($2.15/day, %)": +1.0,  # lower is better
    "Rural Population": +1.0,  # lower is better
    # Disease burden: lower is better
    "HIV Prevalence (15-49)": +1.0,
    "Malaria Incidence": +1.0,
    "Anaemia (women 15-49)": +1.0,
    # Infrastructure: lower travel time is better; higher access/coverage/density better
    "Travel Time to Facility (min)": +1.0,
    "Road Access Index": -1.0,
    "Ambulance Coverage Index": -1.0,
    "Health Facility Density (per 10k)": -1.0,
    # Governance / data: higher is better
    "Gov Health Expenditure (% GDP)": -1.0,
    "Maternal Death Reviews Coverage": -1.0,
    "Reporting Completeness": -1.0,
    "Reporting Timeliness": -1.0,
    "HIS Maturity Index": -1.0,
    "Civil Registration Completeness": -1.0,
    # Shocks: lower is better (i.e., occurrence/intensity/duration increases risk)
    "Epidemic Occurrence (0/1)": +1.0,
    "Conflict Intensity Index": +1.0,
    "Flood Occurrence (0/1)": +1.0,
    "Supply Chain Disruption Index": +1.0,
    "Workforce Strike Duration (days)": +1.0,
}

TARGET_COLUMN = "Maternal Mortality Ratio"

DATA_PATH = "data/afro_maternal_mortality_panel.csv"
MODEL_PATH = "models/maternal_mortality_model.pkl"
METRICS_PATH = "models/model_metrics.json"
