from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.constants import DATA_PATH, FEATURE_COLUMNS, METRICS_PATH, TARGET_COLUMN
from src.predict import load_model, predict_mmr, predict_mmr_interval, scenario_frame

ROOT = Path(__file__).resolve().parents[1]
LATEST_PATH = ROOT / "data" / "afro_country_latest.csv"

DOMAIN_GROUPS: dict[str, list[str]] = {
    "Health system": [
        "Skilled Birth Attendance",
        "ANC4 Coverage",
        "Facility Delivery Rate",
        "Caesarean Section Rate",
        "Emergency Obstetric Care Coverage",
        "Health Workforce Density",
        "Midwife Density",
        "Blood Availability Index",
        "Referral Efficiency Index",
    ],
    "Reproductive": [
        "Fertility Rate",
        "Adolescent Fertility Rate",
        "Contraception (modern)",
    ],
    "Disease": [
        "HIV Prevalence (15-49)",
        "Malaria Incidence",
        "Anaemia (women 15-49)",
    ],
    "Socioeconomic": [
        "Female Literacy",
        "Adult Literacy",
        "Poverty ($2.15/day, %)",
        "Rural Population",
    ],
    "Infrastructure": [
        "Travel Time to Facility (min)",
        "Road Access Index",
        "Ambulance Coverage Index",
        "Health Facility Density (per 10k)",
    ],
    "Governance & data": [
        "Gov Health Expenditure (% GDP)",
        "Civil Registration Completeness",
        "HIS Maturity Index",
        "Maternal Death Reviews Coverage",
        "Reporting Completeness",
        "Reporting Timeliness",
    ],
    "Emergency & shocks": [
        "Epidemic Occurrence (0/1)",
        "Conflict Intensity Index",
        "Flood Occurrence (0/1)",
        "Supply Chain Disruption Index",
        "Workforce Strike Duration (days)",
    ],
}

BOUNDS: dict[str, tuple[float, float]] = {
    # percentages
    "Skilled Birth Attendance": (0.0, 100.0),
    "ANC4 Coverage": (0.0, 100.0),
    "Facility Delivery Rate": (0.0, 100.0),
    "Caesarean Section Rate": (0.0, 50.0),
    "Emergency Obstetric Care Coverage": (0.0, 100.0),
    "Female Literacy": (0.0, 100.0),
    "Adult Literacy": (0.0, 100.0),
    "Contraception (modern)": (0.0, 100.0),
    "Poverty ($2.15/day, %)": (0.0, 100.0),
    "Rural Population": (0.0, 100.0),
    "HIV Prevalence (15-49)": (0.0, 35.0),
    "Anaemia (women 15-49)": (0.0, 80.0),
    "Gov Health Expenditure (% GDP)": (0.0, 15.0),
    "Maternal Death Reviews Coverage": (0.0, 100.0),
    "Reporting Completeness": (0.0, 100.0),
    "Reporting Timeliness": (0.0, 100.0),
    # rates/densities
    "Fertility Rate": (1.0, 8.0),
    "Adolescent Fertility Rate": (0.0, 250.0),
    "Malaria Incidence": (0.0, 800.0),
    "Health Workforce Density": (0.0, 80.0),  # per 10k
    "Midwife Density": (0.0, 40.0),  # per 10k
    "Travel Time to Facility (min)": (0.0, 300.0),
    "Health Facility Density (per 10k)": (0.0, 10.0),
    "Workforce Strike Duration (days)": (0.0, 200.0),
    # indices 0..1
    "Blood Availability Index": (0.0, 1.0),
    "Referral Efficiency Index": (0.0, 1.0),
    "Road Access Index": (0.0, 1.0),
    "Ambulance Coverage Index": (0.0, 1.0),
    "Civil Registration Completeness": (0.0, 100.0),
    "HIS Maturity Index": (0.0, 1.0),
    "Conflict Intensity Index": (0.0, 1.0),
    "Supply Chain Disruption Index": (0.0, 1.0),
    "Epidemic Occurrence (0/1)": (0.0, 1.0),
    "Flood Occurrence (0/1)": (0.0, 1.0),
}


@st.cache_data
def load_panel() -> pd.DataFrame:
    return pd.read_csv(ROOT / DATA_PATH)


@st.cache_data
def load_latest() -> pd.DataFrame:
    if LATEST_PATH.exists():
        return pd.read_csv(LATEST_PATH)
    panel = load_panel()
    complete = panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    idx = complete.groupby("iso3")["year"].idxmax()
    return complete.loc[idx].sort_values("country").reset_index(drop=True)


@st.cache_resource
def get_model():
    return load_model()


def _country_row(df: pd.DataFrame, country: str) -> pd.Series:
    return df.loc[df["country"] == country].iloc[0]


def _preset_scenarios(baseline: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        "Status quo (latest data)": {},
        "High SBA (+15 pp)": {
            "Skilled Birth Attendance": min(100, baseline["Skilled Birth Attendance"] + 15)
        },
        "ANC & facility care (+10 pp each)": {
            "ANC4 Coverage": min(100, baseline.get("ANC4 Coverage", 0) + 10),
            "Facility Delivery Rate": min(100, baseline.get("Facility Delivery Rate", 0) + 10),
        },
        "Lower fertility (-0.5)": {
            "Fertility Rate": max(1.0, baseline["Fertility Rate"] - 0.5)
        },
        "Literacy boost (+10 pp)": {"Female Literacy": min(100, baseline["Female Literacy"] + 10)},
        "Rural health focus (-8 pp rural)": {
            "Rural Population": max(0, baseline["Rural Population"] - 8)
        },
        "Combined package": {
            "Skilled Birth Attendance": min(100, baseline["Skilled Birth Attendance"] + 15),
            "ANC4 Coverage": min(100, baseline.get("ANC4 Coverage", 0) + 10),
            "Contraception (modern)": min(100, baseline.get("Contraception (modern)", 0) + 8),
            "Fertility Rate": max(1.0, baseline["Fertility Rate"] - 0.5),
            "Female Literacy": min(100, baseline["Female Literacy"] + 10),
            "Poverty ($2.15/day, %)": max(0, baseline.get("Poverty ($2.15/day, %)", 0) - 5),
            "Rural Population": max(0, baseline["Rural Population"] - 8),
        },
    }


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def project_to_2030(
    *,
    start_year: int,
    end_year: int,
    baseline: dict[str, float],
    annual_delta: dict[str, float],
) -> pd.DataFrame:
    """Create a yearly projection path for scenario levers.

    Uses linear change per year with plausible bounds.
    """
    years = list(range(int(start_year), int(end_year) + 1))
    rows: list[dict] = []
    for y in years:
        t = y - start_year
        row = {"year": y}
        for col, base in baseline.items():
            d = float(annual_delta.get(col, 0.0))
            lo, hi = BOUNDS.get(col, (-np.inf, np.inf))
            row[col] = _clamp(float(base) + d * t, float(lo), float(hi))
        rows.append(row)
    return pd.DataFrame(rows)


def render_app() -> None:
    model = get_model()
    latest = load_latest()
    countries = latest["country"].tolist()

    st.title("Maternal Mortality What-if Scenarios (WHO AFRO)")
    st.caption(
        "Adjust levers to explore **illustrative** scenario shifts in maternal mortality ratio (MMR). "
        "Predictions start from the country's **observed MMR**; moving a lever changes MMR in the "
        "direction implied on the slider (higher coverage/literacy → lower MMR when marked as protective)."
    )

    tab_whatif, tab_presets, tab_data, tab_about = st.tabs(
        ["Country scenario", "Preset comparisons", "Regional data", "Data & methods"]
    )

    with tab_whatif:
        col_side, col_main = st.columns([1, 2], gap="large")

        with col_side:
            st.subheader("Scenario inputs")
            default_idx = countries.index("Kenya") if "Kenya" in countries else 0
            country = st.selectbox("Country", countries, index=default_idx)
            row = _country_row(latest, country)
            # Baseline for all model features available in the dataset.
            baseline = {c: float(row[c]) for c in FEATURE_COLUMNS if c in row.index}
            observed_mmr = float(row[TARGET_COLUMN])
            data_year = int(row["year"])

            st.info(f"Baseline year: **{data_year}** · Observed MMR: **{observed_mmr:.0f}**")
            st.markdown("Adjust levers below (defaults = latest values).")

            scenario_inputs: dict[str, float] = {}
            for domain, cols in DOMAIN_GROUPS.items():
                available = [c for c in cols if c in baseline]
                if not available:
                    continue
                with st.expander(domain, expanded=(domain in ("Health system", "Reproductive", "Socioeconomic"))):
                    for col in available:
                        lo, hi = BOUNDS.get(col, (float(min(0.0, baseline[col])), float(max(1.0, baseline[col] * 2))))
                        step = 0.1 if (hi - lo) <= 20 else 0.5
                        if col in ("Fertility Rate",):
                            step = 0.05
                        if hi <= 1.0:
                            step = 0.01
                        scenario_inputs[col] = st.slider(
                            col,
                            float(lo),
                            float(hi),
                            float(baseline[col]),
                            float(step),
                        )

            # Only pass model features to the predictor
            model_baseline = {c: float(baseline[c]) for c in FEATURE_COLUMNS if c in baseline}
            model_scenario = {c: float(scenario_inputs.get(c, baseline[c])) for c in FEATURE_COLUMNS if c in baseline}
            baseline_mean, baseline_lo, baseline_hi = predict_mmr_interval(
                model,
                model_baseline,
                anchor=model_baseline,
                anchor_mmr=observed_mmr,
            )

            st.divider()
            st.subheader("Time & projections (to 2030)")
            do_project = st.checkbox("Show projection to 2030", value=True)
            proj_end = 2030
            if do_project:
                st.caption(
                    "Set **annual** changes by domain. Improvements should generally lead to **lower** MMR."
                )
                annual_delta: dict[str, float] = {}
                for domain, cols in DOMAIN_GROUPS.items():
                    available = [c for c in cols if c in baseline]
                    if not available:
                        continue
                    with st.expander(f"Annual change: {domain}", expanded=False):
                        for col in available:
                            default = 0.0
                            if col == "Skilled Birth Attendance":
                                default = 1.5
                            elif col == "Fertility Rate":
                                default = -0.05
                            elif col == "Female Literacy":
                                default = 0.8
                            elif col == "Rural Population":
                                default = -0.1
                            elif col == "Contraception (modern)":
                                default = 0.6
                            elif col == "ANC4 Coverage":
                                default = 1.0
                            elif col == "Poverty ($2.15/day, %)":
                                default = -0.5
                            elif col == "Gov Health Expenditure (% GDP)":
                                default = 0.05
                            elif col == "Malaria Incidence":
                                default = -5.0
                            elif col == "Anaemia (women 15-49)":
                                default = -0.4
                            elif col == "HIV Prevalence (15-49)":
                                default = -0.05

                            step = 0.1
                            if col in ("Fertility Rate",):
                                step = 0.01
                            if col.endswith("(0/1)") or BOUNDS.get(col, (0, 2))[1] <= 1.0:
                                step = 0.01
                            annual_delta[col] = float(
                                st.number_input(f"{col} Δ/year", value=float(default), step=float(step))
                            )

        with col_main:
            predicted_mean, predicted_lo, predicted_hi = predict_mmr_interval(
                model,
                model_scenario,
                anchor=model_baseline,
                anchor_mmr=observed_mmr,
            )
            delta_model = baseline_mean - predicted_mean
            pct_model = (delta_model / baseline_mean * 100) if baseline_mean else 0.0

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Predicted MMR (scenario)",
                f"{predicted_mean:.0f}",
                f"{-delta_model:.0f} vs observed ({pct_model:+.1f}%)",
            )
            m2.metric(
                "Observed MMR (baseline)",
                f"{observed_mmr:.0f}",
                help="Latest MMEIG-aligned estimate from World Bank for this country",
            )
            m3.metric(
                "MMR change (scenario)",
                f"{-delta_model:.0f}",
                f"{pct_model:+.1f}% vs observed",
            )
            st.caption(
                f"Uncertainty (95% interval) — baseline: **{baseline_lo:.0f}–{baseline_hi:.0f}**, "
                f"scenario: **{predicted_lo:.0f}–{predicted_hi:.0f}**. "
                "At default lever settings, scenario equals observed MMR."
            )

            compare_df = pd.DataFrame(
                {
                    "Series": ["Observed (baseline)", "Scenario"],
                    "MMR": [baseline_mean, predicted_mean],
                }
            )
            fig = px.bar(
                compare_df,
                x="Series",
                y="MMR",
                color="Series",
                text="MMR",
                title=f"{country}: maternal mortality ratio comparison",
                color_discrete_sequence=["#3b82f6", "#009E73"],
            )
            fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            fig.update_layout(showlegend=False, yaxis_title="MMR per 100,000 live births")
            st.plotly_chart(fig, use_container_width=True)

            lever_df = pd.DataFrame(
                {
                    "Indicator": [c for c in FEATURE_COLUMNS if c in baseline],
                    "Baseline": [baseline[c] for c in FEATURE_COLUMNS if c in baseline],
                    "Scenario": [model_scenario[c] for c in FEATURE_COLUMNS if c in baseline],
                }
            )
            lever_df["Change"] = lever_df["Scenario"] - lever_df["Baseline"]
            st.subheader("Input changes")
            st.dataframe(lever_df.round(2), use_container_width=True, hide_index=True)

            if do_project:
                proj = project_to_2030(
                    start_year=data_year,
                    end_year=proj_end,
                    baseline=model_baseline,
                    annual_delta=annual_delta,
                )
                means = []
                lo95 = []
                hi95 = []
                for _, r in proj.iterrows():
                    feats = {c: float(r[c]) for c in FEATURE_COLUMNS if c in r.index}
                    m, lo, hi = predict_mmr_interval(
                        model,
                        feats,
                        anchor=model_baseline,
                        anchor_mmr=observed_mmr,
                    )
                    means.append(m)
                    lo95.append(lo)
                    hi95.append(hi)
                proj["mmr_mean"] = means
                proj["mmr_lo95"] = lo95
                proj["mmr_hi95"] = hi95

                st.subheader("Projected MMR trajectory to 2030")
                st.caption(
                    "Band shows 95% interval from bootstrap uncertainty. "
                    "Trajectory applies your annual lever changes relative to observed MMR."
                )

                figp = go.Figure()
                figp.add_trace(
                    go.Scatter(
                        x=proj["year"],
                        y=proj["mmr_mean"],
                        mode="lines+markers",
                        name="MMR (mean)",
                        line=dict(color="#009E73", width=3),
                    )
                )
                figp.add_trace(
                    go.Scatter(
                        x=list(proj["year"]) + list(proj["year"][::-1]),
                        y=list(proj["mmr_hi95"]) + list(proj["mmr_lo95"][::-1]),
                        fill="toself",
                        fillcolor="rgba(0,158,115,0.16)",
                        line=dict(color="rgba(0,0,0,0)"),
                        hoverinfo="skip",
                        name="95% interval",
                    )
                )
                figp.update_layout(
                    xaxis_title="Year",
                    yaxis_title="MMR per 100,000 live births",
                    title=f"{country}: MMR projection ({data_year}–{proj_end})",
                    legend_orientation="h",
                )
                st.plotly_chart(figp, use_container_width=True)

                st.subheader("Projected drivers (model variables) to 2030")
                st.caption(
                    "These are the **input variables** used by the model, projected year-by-year using your annual change settings."
                )
                driver_options = [c for c in FEATURE_COLUMNS if c in proj.columns]
                default_drivers = [
                    c
                    for c in (
                        "Skilled Birth Attendance",
                        "Emergency Obstetric Care Coverage",
                        "Facility Delivery Rate",
                        "Fertility Rate",
                        "Adolescent Fertility Rate",
                        "Contraception (modern)",
                        "ANC4 Coverage",
                        "HIV Prevalence (15-49)",
                        "Malaria Incidence",
                        "Poverty ($2.15/day, %)",
                        "Gov Health Expenditure (% GDP)",
                        "Travel Time to Facility (min)",
                    )
                    if c in driver_options
                ]
                selected_drivers = st.multiselect(
                    "Select variables to plot",
                    options=driver_options,
                    default=default_drivers[:8] if len(default_drivers) > 8 else default_drivers,
                )
                if selected_drivers:
                    long = proj[["year"] + selected_drivers].melt(
                        id_vars=["year"], var_name="Variable", value_name="Value"
                    )
                    dfig = px.line(
                        long,
                        x="year",
                        y="Value",
                        color="Variable",
                        markers=True,
                        title="Projected model inputs (drivers) to 2030",
                    )
                    dfig.update_layout(xaxis_title="Year", yaxis_title="Value", legend_title="Variable")
                    st.plotly_chart(dfig, use_container_width=True)
                else:
                    st.info("Select one or more variables to visualize their projections.")

                proj_out = proj.copy()
                proj_out["Country"] = country
                proj_out["Baseline_Year"] = data_year
                proj_out["Observed_MMR_Reference"] = observed_mmr

                buf2 = BytesIO()
                proj_out.to_excel(buf2, index=False, engine="openpyxl")
                st.download_button(
                    "Download projection to 2030 (Excel)",
                    data=buf2.getvalue(),
                    file_name=f"mmr_projection_{country.replace(' ', '_')}_to_2030.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            export = scenario_frame(model_baseline, model_scenario)
            export["Country"] = country
            export["Baseline_Year"] = data_year
            export["Observed_MMR"] = observed_mmr
            export["Model_Baseline_MMR"] = round(baseline_mean, 1)
            export["Model_Baseline_MMR_Lo95"] = round(baseline_lo, 1)
            export["Model_Baseline_MMR_Hi95"] = round(baseline_hi, 1)
            export["Predicted_MMR"] = round(predicted_mean, 1)
            export["Predicted_MMR_Lo95"] = round(predicted_lo, 1)
            export["Predicted_MMR_Hi95"] = round(predicted_hi, 1)
            export["MMR_Reduction_vs_Model_Baseline"] = round(delta_model, 1)
            export["Pct_Change_vs_Model_Baseline"] = round(pct_model, 2)

            buffer = BytesIO()
            export.to_excel(buffer, index=False, engine="openpyxl")
            st.download_button(
                "Download scenario (Excel)",
                data=buffer.getvalue(),
                file_name=f"mmr_scenario_{country.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with tab_presets:
        st.subheader("Compare policy-style packages")
        preset_country = st.selectbox("Country for presets", countries, key="preset_country")
        prow = _country_row(latest, preset_country)
        pbaseline = {c: float(prow[c]) for c in FEATURE_COLUMNS}
        pobserved = float(prow[TARGET_COLUMN])
        base_pred_mean, _, _ = predict_mmr_interval(
            model,
            pbaseline,
            anchor=pbaseline,
            anchor_mmr=pobserved,
        )

        rows = []
        for name, overrides in _preset_scenarios(pbaseline).items():
            inputs = {**pbaseline, **overrides}
            pred_mean, pred_lo, pred_hi = predict_mmr_interval(
                model,
                inputs,
                anchor=pbaseline,
                anchor_mmr=pobserved,
            )
            rows.append(
                {
                    "Scenario": name,
                    "Predicted MMR": round(pred_mean, 1),
                    "Predicted (Lo–Hi)": f"{pred_lo:.0f}–{pred_hi:.0f}",
                    "Change vs model baseline": round(base_pred_mean - pred_mean, 1),
                    **{
                        f"Δ {k}": round(inputs[k] - pbaseline[k], 2)
                        for k in FEATURE_COLUMNS
                    },
                }
            )
        preset_df = pd.DataFrame(rows)
        st.dataframe(preset_df, use_container_width=True, hide_index=True)

        fig2 = px.bar(
            preset_df,
            x="Scenario",
            y="Predicted MMR",
            title=f"{preset_country}: preset scenario outcomes",
            color="Predicted MMR",
            color_continuous_scale="RdYlGn_r",
        )
        fig2.add_hline(
            y=pobserved, line_dash="dash", line_color="gray", annotation_text="Observed MMR (baseline)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab_data:
        st.subheader("WHO AFRO indicator panel")
        panel = load_panel()
        year_filter = st.slider(
            "Year",
            int(panel["year"].min()),
            int(panel["year"].max()),
            int(panel["year"].max()),
        )
        subset = panel[panel["year"] == year_filter].dropna(subset=[TARGET_COLUMN])
        st.write(f"**{len(subset)}** countries with MMR data in **{year_filter}**")

        map_fig = px.choropleth(
            subset,
            locations="iso3",
            color=TARGET_COLUMN,
            hover_name="country",
            color_continuous_scale="Reds",
            title=f"Maternal mortality ratio by country ({year_filter})",
            labels={TARGET_COLUMN: "MMR"},
        )
        map_fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(map_fig, use_container_width=True)

        scatter_x = st.selectbox("X axis", FEATURE_COLUMNS, index=0)
        scatter = subset.dropna(subset=[scatter_x])
        sc_fig = px.scatter(
            scatter,
            x=scatter_x,
            y=TARGET_COLUMN,
            hover_name="country",
            trendline="ols",
            title=f"{scatter_x} vs MMR ({year_filter})",
        )
        st.plotly_chart(sc_fig, use_container_width=True)
        st.dataframe(subset.sort_values(TARGET_COLUMN, ascending=False), use_container_width=True)

    with tab_about:
        st.markdown(
            """
            ### Data sources (public)
            | Indicator | Source |
            |-----------|--------|
            | Maternal mortality ratio | [World Bank SH.STA.MMRT](https://data.worldbank.org/indicator/SH.STA.MMRT) (UN MMEIG/WHO/UNICEF/UNFPA estimates) |
            | Skilled birth attendance | [SH.STA.BRTW.ZS](https://data.worldbank.org/indicator/SH.STA.BRTW.ZS) |
            | Fertility rate | [SP.DYN.TFRT.IN](https://data.worldbank.org/indicator/SP.DYN.TFRT.IN) |
            | Female literacy | [SE.ADT.LITR.FE.ZS](https://data.worldbank.org/indicator/SE.ADT.LITR.FE.ZS) |
            | Rural population share | [SP.RUR.TOTL.ZS](https://data.worldbank.org/indicator/SP.RUR.TOTL.ZS) |

            Country list: **WHO African Region** member states (47 countries).

            ### Model
            **Monotonic additive model** on log(MMR): each indicator is converted to a
            *risk-oriented* scale (higher = worse), fitted with a **non-negative** coefficient per
            indicator so protective levers cannot increase MMR when moved in the right direction.
            Country scenarios are **anchored** to the observed MMR at baseline lever settings.
            Sparse values are imputed from within-country trends and regional medians; some
            governance/infrastructure/shock variables use **realistic synthetic series** where APIs
            do not publish AFRO-wide annual data.
            Predictions are **illustrative** for planning dialogues—not causal impact estimates.
            """
        )
        metrics_file = ROOT / METRICS_PATH
        if metrics_file.exists():
            st.json(json.loads(metrics_file.read_text(encoding="utf-8")))
