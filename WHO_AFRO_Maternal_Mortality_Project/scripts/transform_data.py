import pandas as pd

raw = pd.read_excel("WHO_AFRO_Maternal_Indicators.xlsx")

pivot = raw.pivot_table(
    index=["country", "year"],
    columns="indicator",
    values="value"
).reset_index()

pivot.columns.name = None

pivot["High_Risk_Score"] = (
    (pivot["Maternal Mortality Ratio"] * 0.4) +
    ((100 - pivot["Skilled Birth Attendance"]) * 0.3) +
    (pivot["Fertility Rate"] * 0.2) +
    (pivot["Rural Population"] * 0.1)
)

pivot.to_excel("WHO_AFRO_Processed_Data.xlsx", index=False)

print("Processed data saved")