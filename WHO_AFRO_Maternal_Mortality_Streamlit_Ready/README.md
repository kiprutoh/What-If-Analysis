# WHO AFRO Maternal Mortality What-if Analysis

Interactive **what-if** scenario tool for the WHO African Region. Adjust skilled birth attendance, fertility, female literacy, and rural population share to explore illustrative changes in **maternal mortality ratio (MMR)**.

Data are pulled from the **World Bank Open Data API** (public indicators aligned with WHO/UN MMEIG maternal mortality estimates).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: refresh data and retrain model
python scripts/build_dataset.py
python scripts/train_model.py

streamlit run app.py
```

Pre-built `data/` and `models/` files are included so the app runs immediately after `pip install` without calling the API.

## App features

- **Country scenario** — sliders vs latest observed values; compare predicted vs observed MMR
- **Preset comparisons** — policy-style bundles (SBA scale-up, literacy, combined package)
- **Regional data** — choropleth and scatter plots over the AFRO panel
- **Excel export** — download scenario inputs and outputs

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select repo, branch `main`, main file **`app.py`**
4. Deploy (uses `requirements.txt` automatically)

## Repository layout

```
app.py                 # Streamlit UI
src/                   # constants, data API, prediction helpers
scripts/
  build_dataset.py     # download World Bank indicators
  train_model.py       # train gradient boosting model
data/                  # CSV panel (committed for deploy)
models/                # trained model + metrics JSON
.streamlit/config.toml   # theme
```

## Data refresh

```bash
python scripts/build_dataset.py   # writes data/afro_maternal_mortality_panel.csv
python scripts/train_model.py     # writes models/maternal_mortality_model.pkl
```

## Disclaimer

Outputs are **scenario illustrations** from a statistical model fit to ecological (country-year) data. They do not replace national MMEIG estimates or causal program evaluation.
