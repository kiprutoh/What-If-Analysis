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
    # Health system / service coverage
    "Skilled Birth Attendance",
    "ANC4 Coverage",
    "Contraception (modern)",
    # Reproductive / demographic
    "Fertility Rate",
    "Adolescent Fertility Rate",
    # Disease burden (risk context)
    "HIV Prevalence (15-49)",
    "Malaria Incidence",
    "Anaemia (women 15-49)",
    # Socioeconomic
    "Female Literacy",
    "Adult Literacy",
    "Poverty ($2.15/day, %)",
    "Rural Population",
    # Governance
    "Gov Health Expenditure (% GDP)",
]

TARGET_COLUMN = "Maternal Mortality Ratio"

DATA_PATH = "data/afro_maternal_mortality_panel.csv"
MODEL_PATH = "models/maternal_mortality_model.pkl"
METRICS_PATH = "models/model_metrics.json"
