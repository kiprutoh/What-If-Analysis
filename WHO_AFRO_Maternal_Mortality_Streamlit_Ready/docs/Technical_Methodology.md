# Technical Background & Methodology
**Project**: WHO AFRO Maternal Mortality What‑If Analysis (Streamlit)  
**Scope**: Country‑level scenario exploration for WHO African Region member states, with projections to **2030**.  
**Primary outcome**: **Maternal Mortality Ratio (MMR)** (maternal deaths per 100,000 live births).  

---

## 1) Background and purpose
Maternal mortality reduction is influenced by interacting drivers across health system readiness, reproductive health, disease burden, socioeconomic conditions, infrastructure access, governance/data systems, and acute shocks (epidemics, conflict, floods, etc.). Decision‑makers often require an interactive way to explore “what if” packages—e.g., improving skilled birth attendance while reducing fertility and strengthening referral systems—and understand how these shifts might relate to lower MMR over time.

This application provides:
- A **scenario sandbox** for adjusting predictors grouped by **domains**.
- A **Bayesian modeling approach** that returns not only a mean prediction but also a **predictive uncertainty interval**.
- A **time factor**: annual change assumptions generate **year‑by‑year projections through 2030**, emphasizing the desirable direction of progress: **reductions in MMR**.

This is designed for **planning and dialogue**; it is not a causal impact evaluator.

---

## 2) Predictor variables (Step 2) and domains
Requested predictors are cataloged and grouped into domains in:
- `src/domains.py`

Domains represented in the application UI include:
- **Health system**
- **Reproductive**
- **Disease**
- **Socioeconomic**
- **Infrastructure**
- **Governance & data**
- **Emergency & shocks**

Where possible, the implementation uses **authoritative global APIs**; where global APIs are incomplete or unavailable, the application uses **realistic synthetic data** to demonstrate the concept (see Section 5).

---

## 3) Data sources and API extraction
The pipeline builds a **country‑year panel** for WHO AFRO countries and merges predictors into a single table.

### 3.1 World Bank Open Data API (primary global panel)
Implementation: `src/data_sources.py`  
Access pattern: `https://api.worldbank.org/v2/country/{ISO3}/indicator/{CODE}?format=json&date=2000:2023`

Extracted indicators include (examples):
- MMR: `SH.STA.MMRT` (MMEIG/WHO/UN agencies estimates via World Bank)
- Skilled birth attendance: `SH.STA.BRTW.ZS`
- Fertility rate: `SP.DYN.TFRT.IN`
- Adolescent fertility: `SP.ADO.TFRT`
- Contraception (modern): `SP.DYN.CONM.ZS`
- ANC4 coverage: `SP.DYN.4ANTE.ZS`
- HIV prevalence (15–49): `SH.DYN.AIDS.ZS`
- Malaria incidence: `SH.MLR.INCD.P3`
- Anaemia (women 15–49): `SH.ANM.ALLW.ZS`
- Poverty ($2.15/day): `SI.POV.DDAY`
- Government health expenditure (% GDP): `SH.XPD.GHED.GD.ZS`
- Female literacy: `SE.ADT.LITR.FE.ZS`
- Adult literacy: `SE.ADT.LITR.ZS`
- Rural population share: `SP.RUR.TOTL.ZS`

### 3.2 WHO Global Health Observatory (GHO) OData API
Client: `src/who_gho.py`  
Base URL: `https://ghoapi.azureedge.net/api/`  
Example endpoint: `https://ghoapi.azureedge.net/api/HWF_0006`

Used to fetch indicators that are often not consistently exposed via the World Bank panel, such as:
- Caesarean section rate (e.g., `WHS4_115`)
- Health workforce densities (e.g., medical doctors `HWF_0003`, nursing & midwifery `HWF_0006`)

These are merged by `iso3` and `year`.

### 3.3 UNICEF SDMX API (via `unicefdata`)
Client wrapper: `src/unicef_sdmx.py`  
Library: `unicefdata` (added in `requirements.txt`)  
API base: `https://sdmx.data.unicef.org/ws/public/sdmxapi/rest/`

Used to pull MNCH indicators where available, for example:
- Institutional delivery: `MNCH_INSTDEL`
- Caesarean section: `MNCH_CSEC`

If UNICEF provides a value for a country‑year where WHO/WB are missing, the pipeline can preferentially fill gaps.

---

## 4) Data processing and panel construction
Implementation: `src/data_sources.py`, `scripts/build_dataset.py`

Steps:
1. **Download** each indicator for WHO AFRO ISO3 countries (2000–2023).
2. **Outer‑join** indicators into a unified panel on `(iso3, country, year)`.
3. **Imputation** for sparse indicators (when enabled):
   - Within‑country **forward/back fill** across years.
   - Remaining gaps filled with **regional median by year**.
4. Add additional authoritative series (WHO GHO, UNICEF SDMX).
5. Apply **synthetic fill** for predictors not available or too sparse for demo.

Artifacts produced:
- `data/afro_maternal_mortality_panel.csv` (country‑year panel)
- `data/afro_country_latest.csv` (latest complete snapshot per country for the app)

---

## 5) Realistic synthetic (dummy) data strategy
Implementation: `src/synthetic_data.py`

Some requested variables are not consistently available globally via open APIs (or require credentials / national systems), e.g.:
- Emergency obstetric care coverage, referral efficiency, blood availability
- Travel time to facility (typically requires GIS/access modeling)
- CRVS completeness and HIS maturity (often from specialized assessments)
- Shocks: conflict intensity (ACLED API key), floods (EM‑DAT access), supply chain disruption, strikes

To demonstrate the scenario concept while keeping values realistic:
- Synthetic series are generated **per country** using a **bounded random walk**:
  - Values constrained to plausible bounds (e.g., 0–100%, 0–1 indices, minutes).
  - Small annual drift + noise to mimic gradual change.
  - **Deterministic seed** by `(iso3, variable)` to ensure reproducibility across runs.
- Binary shock indicators (0/1) are created by thresholding the latent walk.

The synthetic fill is applied **after** authoritative API merges, so real data takes precedence.

---

## 6) Bayesian modeling approach
Implementation: `scripts/train_model.py`, `src/predict.py`

### 6.1 Target transformation
The model predicts **log(MMR)** to ensure positivity and reduce skew:
- \( y = \log(\max(\mathrm{MMR}, 1)) \)

### 6.2 Model choice
The core model is **Bayesian linear regression** using `sklearn.linear_model.BayesianRidge` inside a pipeline:
- Standardization: `StandardScaler`
- Regressor: `BayesianRidge`

This supports:
- A **posterior predictive mean**
- A **predictive standard deviation** (via `predict(return_std=True)`), enabling uncertainty intervals.

### 6.3 Predictive interval calculation
Implementation: `predict_mmr_interval()` in `src/predict.py`

Because the model predicts log(MMR), intervals are computed on the log scale and exponentiated:
- \( \mu, \sigma = \text{predict}(X, \text{return\_std=True}) \)
- 95% interval: \( \exp(\mu \pm 1.96\sigma) \)
- Mean on original scale approximated by log‑normal correction: \( \exp(\mu + 0.5\sigma^2) \)

---

## 7) Scenario engine and projections to 2030
Implementation: `src/app_page.py`

### 7.1 Cross‑domain scenarios
The UI groups predictors by domain and allows users to adjust:
- Current‑year scenario values
- **Annual change rates** per predictor (domain‑aware)

### 7.2 Time projection
Given:
- baseline year \(t_0\) (latest data year for a country)
- annual deltas per predictor

The app generates a yearly path:
- \( x_{t} = \mathrm{clamp}(x_{t_0} + \Delta x \cdot (t - t_0)) \)
for \(t \in [t_0, 2030]\), with plausible bounds per predictor.

For each year, the app computes:
- MMR mean projection
- 95% predictive interval band

Exports:
- Year‑by‑year projection table (Excel)
- Scenario snapshot (Excel)

---

## 8) Deployment and reproducibility
- The app is packaged for Streamlit deployment:
  - Entry point: `app.py` (delegates to `index.py`)
  - Main UI: `index.py` + `src/app_page.py`
- Data/model artifacts are stored in‑repo for deploy:
  - `data/*.csv`
  - `models/*.pkl`, `models/model_metrics.json`

Rebuild commands:
```bash
python scripts/build_dataset.py
python scripts/train_model.py
streamlit run app.py
```

---

## 9) Limitations and interpretation
- **Not causal**: the model is a statistical association model at country‑year level.
- Synthetic data are included for demonstration and should be replaced with national sources (DHIS2/HMIS/CRVS, facility assessments) where possible.
- Uncertainty intervals reflect model predictive uncertainty given data and model assumptions, not full structural uncertainty.

---

## 10) Implementation map (where to look in code)
- Data catalog/domains: `src/domains.py`
- World Bank extraction + panel build: `src/data_sources.py`
- WHO GHO client: `src/who_gho.py`
- UNICEF SDMX client: `src/unicef_sdmx.py`
- Synthetic data generator: `src/synthetic_data.py`
- Model training: `scripts/train_model.py`
- Prediction utilities: `src/predict.py`
- Scenario UI + 2030 projection: `src/app_page.py`

