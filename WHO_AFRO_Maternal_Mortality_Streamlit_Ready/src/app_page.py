from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.constants import DATA_PATH, FEATURE_COLUMNS, METRICS_PATH, TARGET_COLUMN
from src.predict import load_model, predict_mmr, predict_mmr_interval, scenario_frame

ROOT = Path(__file__).resolve().parents[1]
LATEST_PATH = ROOT / "data" / "afro_country_latest.csv"


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
        "Lower fertility (-0.5)": {
            "Fertility Rate": max(1.0, baseline["Fertility Rate"] - 0.5)
        },
        "Literacy boost (+10 pp)": {"Female Literacy": min(100, baseline["Female Literacy"] + 10)},
        "Rural health focus (-8 pp rural)": {
            "Rural Population": max(0, baseline["Rural Population"] - 8)
        },
        "Combined package": {
            "Skilled Birth Attendance": min(100, baseline["Skilled Birth Attendance"] + 15),
            "Fertility Rate": max(1.0, baseline["Fertility Rate"] - 0.5),
            "Female Literacy": min(100, baseline["Female Literacy"] + 10),
            "Rural Population": max(0, baseline["Rural Population"] - 8),
        },
    }


def render_app() -> None:
    model = get_model()
    latest = load_latest()
    countries = latest["country"].tolist()

    st.title("Maternal Mortality What-if Scenarios (WHO AFRO)")
    st.caption(
        "Adjust levers to explore **illustrative** scenario shifts in maternal mortality ratio (MMR). "
        "Observed MMR is shown as reference; deltas are computed vs the model baseline."
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
            baseline = {c: float(row[c]) for c in FEATURE_COLUMNS}
            observed_mmr = float(row[TARGET_COLUMN])
            data_year = int(row["year"])

            st.info(f"Baseline year: **{data_year}** · Observed MMR: **{observed_mmr:.0f}**")
            st.markdown("Adjust levers below (defaults = latest observed values).")

            sba = st.slider(
                "Skilled birth attendance (%)",
                0.0,
                100.0,
                baseline["Skilled Birth Attendance"],
                0.5,
            )
            fertility = st.slider(
                "Fertility rate (births per woman)",
                1.0,
                8.0,
                baseline["Fertility Rate"],
                0.1,
            )
            literacy = st.slider(
                "Female literacy (%)", 0.0, 100.0, baseline["Female Literacy"], 0.5
            )
            rural = st.slider(
                "Rural population (%)", 0.0, 100.0, baseline["Rural Population"], 0.5
            )

            scenario_inputs = {
                "Skilled Birth Attendance": sba,
                "Fertility Rate": fertility,
                "Female Literacy": literacy,
                "Rural Population": rural,
            }
            baseline_mean, baseline_lo, baseline_hi = predict_mmr_interval(model, baseline)

        with col_main:
            predicted_mean, predicted_lo, predicted_hi = predict_mmr_interval(model, scenario_inputs)
            delta_model = baseline_mean - predicted_mean
            pct_model = (delta_model / baseline_mean * 100) if baseline_mean else 0.0

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Predicted MMR (scenario)",
                f"{predicted_mean:.0f}",
                f"{-delta_model:.0f} vs model baseline",
            )
            m2.metric("Model at baseline inputs", f"{baseline_mean:.0f}")
            m3.metric(
                "Observed MMR (reference)",
                f"{observed_mmr:.0f}",
                help="Latest MMEIG-aligned estimate from World Bank for this country",
            )
            st.caption(
                f"Uncertainty (95% interval) — baseline: **{baseline_lo:.0f}–{baseline_hi:.0f}**, "
                f"scenario: **{predicted_lo:.0f}–{predicted_hi:.0f}**."
            )

            compare_df = pd.DataFrame(
                {
                    "Series": ["Model baseline", "Scenario", "Observed (reference)"],
                    "MMR": [baseline_mean, predicted_mean, observed_mmr],
                }
            )
            fig = px.bar(
                compare_df,
                x="Series",
                y="MMR",
                color="Series",
                text="MMR",
                title=f"{country}: maternal mortality ratio comparison",
                color_discrete_sequence=["#6c757d", "#009E73", "#3b82f6"],
            )
            fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            fig.update_layout(showlegend=False, yaxis_title="MMR per 100,000 live births")
            st.plotly_chart(fig, use_container_width=True)

            lever_df = pd.DataFrame(
                {
                    "Indicator": FEATURE_COLUMNS,
                    "Baseline": [baseline[c] for c in FEATURE_COLUMNS],
                    "Scenario": [scenario_inputs[c] for c in FEATURE_COLUMNS],
                }
            )
            lever_df["Change"] = lever_df["Scenario"] - lever_df["Baseline"]
            st.subheader("Input changes")
            st.dataframe(lever_df.round(2), use_container_width=True, hide_index=True)

            export = scenario_frame(baseline, scenario_inputs)
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
        base_pred_mean, _, _ = predict_mmr_interval(model, pbaseline)

        rows = []
        for name, overrides in _preset_scenarios(pbaseline).items():
            inputs = {**pbaseline, **overrides}
            pred_mean, pred_lo, pred_hi = predict_mmr_interval(model, inputs)
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
        fig2.add_hline(y=base_pred_mean, line_dash="dash", annotation_text="Model baseline")
        fig2.add_hline(
            y=pobserved, line_dash="dot", line_color="gray", annotation_text="Observed MMR"
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
            Ridge regression on log(MMR) using country-year panel (2000–2023).
            Sparse literacy/rural values are imputed from within-country trends and regional medians.
            Predictions are **illustrative** for planning dialogues—not causal impact estimates.
            """
        )
        metrics_file = ROOT / METRICS_PATH
        if metrics_file.exists():
            st.json(json.loads(metrics_file.read_text(encoding="utf-8")))
