import pandas as pd
import requests

AFRO_COUNTRIES = [
    "Angola", "Benin", "Botswana", "Burkina Faso",
    "Burundi", "Cameroon", "Chad", "Congo",
    "Democratic Republic of Congo", "Ethiopia",
    "Eswatini", "Ghana", "Kenya", "Lesotho",
    "Liberia", "Madagascar", "Malawi", "Mali",
    "Mozambique", "Namibia", "Niger", "Nigeria",
    "Rwanda", "Senegal", "Sierra Leone",
    "South Africa", "South Sudan", "Tanzania",
    "Uganda", "Zambia", "Zimbabwe"
]

INDICATORS = {
    "SH.STA.MMRT": "Maternal Mortality Ratio",
    "SH.STA.BRTC.ZS": "Skilled Birth Attendance",
    "SP.DYN.TFRT.IN": "Fertility Rate",
    "SP.RUR.TOTL.ZS": "Rural Population",
    "SE.ADT.LITR.FE.ZS": "Female Literacy"
}

BASE_URL = "https://api.worldbank.org/v2/country/{}/indicator/{}?format=json&per_page=100"

all_data = []

for country in AFRO_COUNTRIES:
    for indicator_code, indicator_name in INDICATORS.items():

        country_code = country[:3].upper()

        url = BASE_URL.format(country_code, indicator_code)

        try:
            response = requests.get(url)
            data = response.json()

            if len(data) > 1:
                for item in data[1]:
                    all_data.append({
                        "country": country,
                        "indicator": indicator_name,
                        "year": item["date"],
                        "value": item["value"]
                    })

        except Exception as e:
            print(country, indicator_name, e)

df = pd.DataFrame(all_data)

with pd.ExcelWriter("WHO_AFRO_Maternal_Indicators.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Raw_Data", index=False)

print("Excel file created")