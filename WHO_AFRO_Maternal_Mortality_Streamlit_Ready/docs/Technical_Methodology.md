# Technical Background & Methodology
**Project**: WHO AFRO Maternal Mortality What‑If Analysis (Streamlit)  
**Scope**: Country‑level scenario exploration for WHO African Region member states, with projections to **2030**.  
**Primary outcome**: **Maternal Mortality Ratio (MMR)** (maternal deaths per 100,000 live births).  
**Document version**: May 2026 — includes equations and platform user steps.

---

## 1) Background and purpose

Maternal mortality reduction is influenced by interacting drivers across health system readiness, reproductive health, disease burden, socioeconomic conditions, infrastructure access, governance and data systems, and acute shocks (epidemics, conflict, floods, etc.). Decision‑makers often require an interactive way to explore “what if” packages—for example improving skilled birth attendance and ANC4 coverage while increasing government health expenditure—and understand how these shifts might relate to lower MMR over time.

This application provides:

- A **scenario sandbox** for adjusting predictors grouped by **domains**.
- A **monotonic, direction-aware modeling approach** (protective vs risk indicators) with **bootstrap uncertainty intervals**.
- A **time factor**: annual change assumptions generate **year‑by‑year projections through 2030**, emphasizing reductions in MMR.

This is designed for **planning and dialogue**; it is not a causal impact evaluator.

---

## 2) Predictor variables and domains

Predictors are cataloged in `src/domains.py` and grouped in the UI as:

- **Health system** (skilled birth attendance, ANC4, facility delivery, EmOC, workforce, blood, referral, etc.)
- **Reproductive** (contraception, fertility, adolescent fertility)
- **Disease** (HIV, malaria, anaemia)
- **Socioeconomic** (literacy, poverty, rural population)
- **Infrastructure** (travel time, road access, ambulance, facility density)
- **Governance & data** (government health expenditure, CRVS, HIS maturity, reporting)
- **Emergency & shocks** (epidemics, conflict, floods, supply chain, strikes)

Where global APIs are incomplete, **realistic synthetic data** fill gaps for demonstration (Section 5).

---

## 3) Data sources and API extraction

The pipeline builds a **country‑year panel** for WHO AFRO countries and merges predictors into a single table.

### 3.1 World Bank Open Data API (primary global panel)

Implementation: `src/data_sources.py`  
Access pattern: `https://api.worldbank.org/v2/country/{ISO3}/indicator/{CODE}?format=json&date=2000:2023`

Key indicators include:

- MMR: `SH.STA.MMRT` (UN MMEIG estimates via World Bank)
- Skilled birth attendance: `SH.STA.BRTW.ZS`
- ANC4 coverage: `SP.DYN.4ANTE.ZS`
- Government health expenditure (% GDP): `SH.XPD.GHED.GD.ZS`
- Contraception (modern): `SP.DYN.CONM.ZS`
- Fertility, adolescent fertility, HIV, malaria, anaemia, poverty, literacy, rural population (see `src/constants.py`)

### 3.2 WHO Global Health Observatory (GHO) OData API

Client: `src/who_gho.py` — caesarean section rate, doctor and nursing/midwifery densities.

### 3.3 UNICEF SDMX API (via `unicefdata`)

Client: `src/unicef_sdmx.py` — institutional delivery, caesarean section where available.

---

## 4) Data processing and panel construction

Implementation: `src/data_sources.py`, `scripts/build_dataset.py`

1. Download each indicator for WHO AFRO ISO3 countries (2000–2023).
2. Outer‑join indicators on `(iso3, country, year)`.
3. Impute sparse series: within‑country forward/back fill, then regional median by year.
4. Merge WHO GHO and UNICEF series.
5. Apply synthetic fill only where authoritative data are missing.

**Outputs:**

- `data/afro_maternal_mortality_panel.csv`
- `data/afro_country_latest.csv`

---

## 5) Realistic synthetic data strategy

Implementation: `src/synthetic_data.py`

Variables not available as open annual AFRO-wide series (EmOC coverage, referral efficiency, travel time, conflict, floods, etc.) use **bounded random walks** with deterministic seeds per `(iso3, variable)`. Synthetic values never overwrite World Bank, WHO, or UNICEF observations.

---

## 6) Statistical methods and equations

Implementation: `scripts/train_model.py`, `src/predict.py`, `src/constants.py`

### 6.1 Notation

| Symbol | Meaning |
|--------|---------|
| \( i \) | Country index (ISO3) |
| \( t \) | Year |
| \( j \) | Predictor index, \( j = 1, \ldots, J \) (\( J = 34 \)) |
| \( \mathrm{MMR}_{it} \) | Observed maternal mortality ratio |
| \( x_{ijt} \) | Raw value of predictor \( j \) for country \( i \), year \( t \) |
| \( m_j \) | Direction multiplier: \( m_j = -1 \) (protective), \( m_j = +1 \) (risk) |

### 6.2 Target transformation

The model works on the log scale to enforce positivity and reduce skew:

\[
y_{it} = \log\!\left(\max(\mathrm{MMR}_{it},\, 1)\right)
\]

### 6.3 Risk-oriented predictors

Each raw indicator is mapped to a **risk-oriented** value where higher always implies higher expected MMR:

\[
\tilde{x}_{ijt} = m_j \cdot x_{ijt}
\]

**Examples:**

- ANC4 coverage (protective): \( m_j = -1 \). If ANC4 rises, \( \tilde{x} \) falls → predicted MMR falls.
- Government health expenditure % GDP (protective): \( m_j = -1 \). Higher spending → lower \( \tilde{x} \) → lower MMR.
- Fertility rate (risk): \( m_j = +1 \). Higher fertility → higher \( \tilde{x} \) → higher MMR.

### 6.4 Standardization

For training, each risk-oriented column is standardized using panel mean and standard deviation:

\[
z_{ijt} = \frac{\tilde{x}_{ijt} - \mu_j}{\sigma_j}
\]

where \( \mu_j \) and \( \sigma_j \) are computed over all country‑year rows in the training panel.

### 6.5 Monotonic additive model (training)

Each predictor receives a **non-negative** coefficient \( \beta_j \geq 0 \), fitted from its **univariate** relationship with \( y_{it} \) on standardized inputs. This avoids joint multicollinearity zeroing coefficients when many indicators move together.

Structural linear predictor on the log scale:

\[
\eta_{it} = \beta_0 + \sum_{j=1}^{J} \beta_j \, z_{ijt}, \qquad \beta_j \geq 0
\]

In-sample panel prediction (before country anchoring):

\[
\widehat{\mathrm{MMR}}_{it} = \exp(\eta_{it})
\]

**Coefficient shrinkage:** univariate coefficients are scaled by a factor \( \lambda \in (0,1] \) chosen to balance panel mean absolute error while preserving lever responsiveness.

### 6.6 Country-anchored scenario prediction (platform)

For interactive scenarios, let \( \mathbf{x}_i^{(0)} \) be the **baseline** lever vector (latest country snapshot) and \( \mathbf{x}_i^{(1)} \) the **scenario** vector chosen by the user. Let \( \mathrm{MMR}_i^{\mathrm{obs}} \) be the observed MMR at baseline.

Define structural scores (using the same \( \beta \) and standardization as training):

\[
\eta_i^{(0)} = \beta_0 + \sum_{j=1}^{J} \beta_j \, z_{ij}^{(0)}, \qquad
\eta_i^{(1)} = \beta_0 + \sum_{j=1}^{J} \beta_j \, z_{ij}^{(1)}
\]

**Anchored scenario MMR** (ensures default sliders reproduce observed MMR):

\[
\log \widehat{\mathrm{MMR}}_i^{(1)} =
\log\!\left(\mathrm{MMR}_i^{\mathrm{obs}}\right) +
\gamma \left( \eta_i^{(1)} - \eta_i^{(0)} \right)
\]

\[
\widehat{\mathrm{MMR}}_i^{(1)} = \exp\!\left( \log \widehat{\mathrm{MMR}}_i^{(1)} \right)
\]

where \( \gamma \) is the **scenario log gain** (`scenario_log_gain`), calibrated so a +15 percentage-point increase in skilled birth attendance yields approximately a 6% median MMR reduction across countries.

**Directional property:** for a protective indicator with \( m_j = -1 \), increasing \( x_j \) decreases \( z_j \), decreases \( \eta \), and therefore decreases anchored \( \widehat{\mathrm{MMR}} \) when \( \gamma > 0 \).

### 6.7 Bootstrap uncertainty intervals

Bootstrap resampling ( \( B = 120 \) ) refits univariate positive coefficients on resampled country‑year rows. For each bootstrap draw \( b \):

\[
\eta_{i,b}^{(1)} = \beta_{0,b} + \sum_{j=1}^{J} \beta_{j,b} \, z_{ij}^{(1)}
\]

Anchored bootstrap MMR samples:

\[
\widehat{\mathrm{MMR}}_{i,b}^{(1)} =
\exp\!\left(
\log\!\left(\mathrm{MMR}_i^{\mathrm{obs}}\right) +
\gamma \left( \eta_{i,b}^{(1)} - \eta_{i,b}^{(0)} \right)
\right)
\]

**95% interval** (percentile method):

\[
\left[
Q_{0.025}\!\left(\widehat{\mathrm{MMR}}_{i,b}^{(1)}\right),\;
Q_{0.975}\!\left(\widehat{\mathrm{MMR}}_{i,b}^{(1)}\right)
\right]
\]

**Point estimate:** mean of bootstrap samples \( \frac{1}{B}\sum_b \widehat{\mathrm{MMR}}_{i,b}^{(1)} \).

### 6.8 Time projection of inputs (2000–2030)

Given baseline year \( t_0 \), annual change \( \Delta_j \) per predictor, and bounds \( [L_j, U_j] \):

\[
x_{ijt} = \mathrm{clamp}\!\left( x_{ijt_0} + \Delta_j \cdot (t - t_0),\; L_j,\; U_j \right),
\qquad t \in [t_0,\, 2030]
\]

where:

\[
\mathrm{clamp}(v, L, U) = \min\!\left(U,\, \max(L,\, v)\right)
\]

For each year \( t \), \( \widehat{\mathrm{MMR}}_{it} \) is computed using Equations (6)–(7) with \( \mathbf{x}_{it} \) from Equation (9).

### 6.9 Methods narrative (summary paragraphs)

This application supports illustrative what-if planning for maternal mortality in the WHO African Region using publicly reported MMR and harmonised predictors. The data layer combines World Bank, WHO GHO, and UNICEF APIs with imputation and targeted synthetic series. Indicator directionality is enforced through risk multipliers \( m_j \). The statistical engine uses a monotonic additive log-linear model with country anchoring to observed MMR and bootstrap intervals. Projections to 2030 apply linear annual changes to inputs with bounded trajectories. Results are associative and illustrative—not causal impact estimates.

---

## 7) Step-by-step guide: using the platform

The Streamlit application entry point is `app.py` (landing page in `index.py`; analysis in `src/app_page.py`).

### Step 1 — Open the application

- **Local:** run `streamlit run app.py` in the project folder.
- **Deployed:** open the Streamlit Cloud URL for the repository.

The **Home** page describes the tool and links to the analysis workspace.

### Step 2 — Navigate to What-if analysis

In the **sidebar**, select **What-if analysis** (or click **Launch what-if analysis** on the Home page).

### Step 3 — Choose a country

On the **Country scenario** tab, use the **Country** dropdown. The app loads the latest complete snapshot for that country and displays:

- **Baseline year** (data year for levers)
- **Observed MMR** (reference from World Bank / MMEIG)

### Step 4 — Adjust scenario levers (by domain)

Expand domain sections in the left panel (**Health system**, **Reproductive**, etc.):

- **Green (🟢) sliders:** protective indicators—moving **up** should **lower** projected MMR (e.g. ANC4, government health expenditure, skilled birth attendance).
- **Orange (🟠) sliders:** risk indicators—moving **up** should **raise** projected MMR (e.g. malaria incidence, conflict index).

Hover over a slider for a direction hint. Defaults equal the latest observed values.

### Step 5 — Read scenario results

The right panel shows:

1. **Predicted MMR (scenario)** vs **Observed MMR (baseline)**
2. **MMR change** (absolute and % vs observed)
3. **Bar chart** comparing observed baseline and scenario
4. **Input changes table** with column **If you increase this lever** (MMR should ↓ or ↑)

At default settings, scenario MMR equals observed MMR.

### Step 6 — Configure projections to 2030 (optional)

Enable **Show projection to 2030**. Under **Annual change: [domain]**, enter **Δ/year** for each indicator—for example:

- ANC4: +1.0 percentage points per year
- Government health expenditure: +0.05 percentage points of GDP per year
- Malaria incidence: −5.0 per year (reducing burden)

Positive Δ on green indicators generally drives **downward** MMR trajectories.

### Step 7 — Review projection charts

- **Projected MMR trajectory to 2030** — mean line plus 95% bootstrap band
- **Projected drivers** — multiselect variables to plot input paths

### Step 8 — Export results

- **Download scenario (Excel)** — baseline vs scenario levers and MMR with intervals
- **Download projection to 2030 (Excel)** — year-by-year MMR and drivers

### Step 9 — Compare preset packages

Open the **Preset comparisons** tab, select a country, and review packaged scenarios (e.g. high SBA, combined package). Bars show predicted MMR vs observed baseline.

### Step 10 — Explore regional data

The **Regional data** tab provides a choropleth map, scatter plots, and tabular MMR by country and year.

### Step 11 — Read methods and model diagnostics

The **Data & methods** tab contains the full methods narrative and a JSON block of model metrics (`models/model_metrics.json`).

---

## 8) Scenario engine (implementation reference)

Implementation: `src/app_page.py`

- Cross-domain sliders and annual Δ inputs
- `project_to_2030()` implements Equation (9)
- `predict_mmr_interval()` implements Equations (6)–(8)
- Excel export via `openpyxl`

---

## 9) Deployment and reproducibility

- Entry: `app.py` → `index.py` → `src/app_page.py`
- Artifacts: `data/*.csv`, `models/maternal_mortality_model.pkl`, `models/model_metrics.json`

**Rebuild pipeline:**

```bash
python scripts/build_dataset.py
python scripts/train_model.py
python scripts/generate_methodology_doc.py
streamlit run app.py
```

---

## 10) Limitations and interpretation

- **Not causal:** country-year associations only; no programme attribution.
- **Synthetic inputs** should be replaced with national HMIS/CRVS/facility data where possible.
- **Calibration** (`γ`, `λ`) improves scenario responsiveness for planning; it is not epidemiological forecasting.
- **Uncertainty bands** reflect bootstrap sampling given the model form, not all structural uncertainty.

---

## 11) Implementation map

| Component | File |
|-----------|------|
| Domains catalog | `src/domains.py` |
| World Bank panel | `src/data_sources.py` |
| WHO GHO client | `src/who_gho.py` |
| UNICEF client | `src/unicef_sdmx.py` |
| Synthetic data | `src/synthetic_data.py` |
| Model training | `scripts/train_model.py` |
| Prediction | `src/predict.py` |
| Scenario UI | `src/app_page.py` |
| Methods text | `src/methods_narrative.py` |
| Word export | `scripts/generate_methodology_doc.py` |
