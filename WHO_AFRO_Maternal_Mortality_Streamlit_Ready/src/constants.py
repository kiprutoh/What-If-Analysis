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
    "literacy": "SE.ADT.LITR.FE.ZS",  # Literacy rate, adult female (%)
    "rural": "SP.RUR.TOTL.ZS",      # Rural population (% of total)
}

FEATURE_COLUMNS = [
    "Skilled Birth Attendance",
    "Fertility Rate",
    "Female Literacy",
    "Rural Population",
]

TARGET_COLUMN = "Maternal Mortality Ratio"

DATA_PATH = "data/afro_maternal_mortality_panel.csv"
MODEL_PATH = "models/maternal_mortality_model.pkl"
METRICS_PATH = "models/model_metrics.json"
