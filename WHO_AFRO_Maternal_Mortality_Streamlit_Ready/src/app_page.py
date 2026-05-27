from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.constants import (
    DATA_PATH,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
    indicator_direction_hint,
    is_protective_indicator,
)
from src.inverse_scenario import PRIORITY_INDICATORS, solve_target_mmr
from src.methods_narrative import methods_markdown
from src.predict import load_model, predict_mmr_interval, scenario_frame
from src.scenario_pane import inject_left_pane_css, render_left_pane_header

ROOT = Path(__file__).resolve().parents[1]
LATEST_PATH = ROOT / "data" / "afro_country_latest.csv"
APP_SECTIONS = [
    "Country scenario",
    "Preset comparisons",
    "Regional data",
    "Data & methods",
]
PRESET_PACKAGE_COUNT = 7
METHODOLOGY_DOCX = ROOT / "docs" / "Technical_Methodology.docx"

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
def get_model(_model_mtime: float):
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


def _default_annual_delta(col: str) -> float:
    defaults = {
        "Skilled Birth Attendance": 1.5,
        "Fertility Rate": -0.05,
        "Female Literacy": 0.8,
        "Rural Population": -0.1,
        "Contraception (modern)": 0.6,
        "ANC4 Coverage": 1.0,
        "Poverty ($2.15/day, %)": -0.5,
        "Gov Health Expenditure (% GDP)": 0.05,
        "Malaria Incidence": -5.0,
        "Anaemia (women 15-49)": -0.4,
        "HIV Prevalence (15-49)": -0.05,
    }
    return float(defaults.get(col, 0.0))


def _apply_chart_data_labels(fig, *, y_format: str = ".0f", textposition: str = "outside") -> None:
    """Show numeric data labels on charts with zero decimal places."""
    fig.update_traces(
        texttemplate=f"%{{y:{y_format}}}",
        textposition=textposition,
    )
    if hasattr(fig, "data") and fig.data:
        first = fig.data[0]
        if getattr(first, "text", None) is not None:
            fig.update_traces(texttemplate=f"%{{text:{y_format}}}")
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")


def _init_scenario_session(country: str) -> int:
    if "scenario_reset_ver" not in st.session_state:
        st.session_state.scenario_reset_ver = 0
    if st.session_state.get("_last_scenario_country") != country:
        st.session_state.scenario_reset_ver += 1
        st.session_state._last_scenario_country = country
        st.session_state.status_quo_confirmed = True
    return int(st.session_state.scenario_reset_ver)


def _reset_scenario_to_status_quo() -> None:
    st.session_state.scenario_reset_ver = int(st.session_state.get("scenario_reset_ver", 0)) + 1
    st.session_state.status_quo_confirmed = True
    st.session_state.target_apply_country = None
    st.session_state.target_apply_values = None


def _slider_default(
    country: str,
    col: str,
    baseline: dict[str, float],
) -> float:
    if (
        st.session_state.get("target_apply_country") == country
        and isinstance(st.session_state.get("target_apply_values"), dict)
    ):
        applied = st.session_state.target_apply_values
        if col in applied:
            return float(applied[col])
    return float(baseline[col])


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
    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom:0.5rem;">
              <div style="font-size:0.7rem;letter-spacing:0.08em;text-transform:uppercase;color:#94a3b8;">
                WHO AFRO
              </div>
              <div style="font-size:1.15rem;font-weight:800;line-height:1.2;color:#f8fafc;">
                Maternal Mortality<br/>
                <span style="background:linear-gradient(90deg,#f87171,#fb923c);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                  Scenario Explorer
                </span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("← Home", use_container_width=True, key="app_home"):
            st.session_state["_mode"] = "Home"
            st.rerun()
        st.markdown("---")
        st.markdown("### Quick guide")
        st.markdown(
            "- **Reset** to status quo first\n"
            "- **Target MMR** → work backwards\n"
            "- 🟢 lever ↑ → lower MMR\n"
            "- **Apply** loads reverse solution"
        )

    model_path = ROOT / MODEL_PATH
    model_mtime = model_path.stat().st_mtime if model_path.exists() else 0.0
    model = get_model(model_mtime)
    latest = load_latest()
    countries = latest["country"].tolist()

    st.title("Maternal Mortality What-if Scenarios (WHO AFRO)")
    st.caption(
        "Adjust levers to explore **illustrative** scenario shifts in maternal mortality ratio (MMR). "
        "Predictions start from the country's **observed MMR**; moving a lever changes MMR in the "
        "direction implied on the slider (higher coverage/literacy → lower MMR when marked as protective)."
    )

    section_default = st.session_state.get("_app_section", APP_SECTIONS[0])
    if section_default not in APP_SECTIONS:
        section_default = APP_SECTIONS[0]
    section = st.radio(
        "Workspace section",
        APP_SECTIONS,
        index=APP_SECTIONS.index(section_default),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state["_app_section"] = section

    if section == "Country scenario":
        inject_left_pane_css()
        col_side, col_main = st.columns([1, 2], gap="large")

        with col_side:
            render_left_pane_header()
            default_idx = countries.index("Kenya") if "Kenya" in countries else 0
            country = st.selectbox("Country", countries, index=default_idx)
            reset_ver = _init_scenario_session(country)
            row = _country_row(latest, country)
            baseline = {c: float(row[c]) for c in FEATURE_COLUMNS if c in row.index}
            observed_mmr = float(row[TARGET_COLUMN])
            data_year = int(row["year"])
            target_result_key = f"target_result_{country}"

            with st.container(border=True):
                st.markdown("**Status quo baseline**")
                st.caption(
                    "Reset all levers to the country's **current status quo** before forward or reverse scenarios."
                )
                st.markdown(
                    f"**{country}** · **{data_year}** · MMR **{observed_mmr:.0f}** · "
                    f"SBA **{baseline.get('Skilled Birth Attendance', 0):.0f}%** · "
                    f"ANC4 **{baseline.get('ANC4 Coverage', 0):.0f}%** · "
                    f"Gov spend **{baseline.get('Gov Health Expenditure (% GDP)', 0):.2f}%** GDP"
                )
                if st.button("Reset to status quo defaults", use_container_width=True, type="primary"):
                    _reset_scenario_to_status_quo()
                    st.rerun()
                st.session_state.status_quo_confirmed = st.checkbox(
                    "Status quo reviewed — proceed with scenarios",
                    value=st.session_state.get("status_quo_confirmed", True),
                )

            st.markdown(
                '<div class="target-panel"><h4>🎯 Target MMR — work backwards</h4></div>',
                unsafe_allow_html=True,
            )
            default_target = max(1, int(round(observed_mmr * 0.75)))
            target_mmr = float(
                st.number_input(
                    "Target MMR (per 100,000 live births)",
                    min_value=1.0,
                    max_value=5000.0,
                    value=float(st.session_state.get(f"target_mmr_input_{country}", default_target)),
                    step=1.0,
                    help="Set a goal MMR; the model estimates indicator values needed from status quo.",
                )
            )
            st.session_state[f"target_mmr_input_{country}"] = target_mmr
            priority_only = st.checkbox(
                "Use priority policy levers only",
                value=True,
                help="Health system, governance, reproductive and key socioeconomic indicators.",
            )
            calc_col, apply_col = st.columns(2)
            with calc_col:
                run_inverse = st.button(
                    "Calculate required indicators",
                    use_container_width=True,
                    type="secondary",
                )
            with apply_col:
                apply_inverse = st.button(
                    "Apply to levers",
                    use_container_width=True,
                    help="Load calculated values into the sliders below.",
                )

            target_result_key = f"target_result_{country}"
            if run_inverse:
                adj = list(PRIORITY_INDICATORS) if priority_only else list(baseline.keys())
                st.session_state[target_result_key] = solve_target_mmr(
                    model,
                    baseline,
                    observed_mmr,
                    target_mmr,
                    BOUNDS,
                    adjustable=adj,
                )
            if apply_inverse and target_result_key in st.session_state:
                st.session_state.target_apply_country = country
                st.session_state.target_apply_values = dict(
                    st.session_state[target_result_key].solution
                )
                st.session_state.scenario_reset_ver = int(
                    st.session_state.get("scenario_reset_ver", 0)
                ) + 1
                st.rerun()

            if target_result_key in st.session_state:
                tr = st.session_state[target_result_key]
                st.caption(tr.message)
                st.metric("Achieved MMR (model)", f"{tr.achieved_mmr:.0f}", f"Target {tr.target_mmr:.0f}")

            st.markdown("---")
            st.markdown("**Forward scenario levers**")
            st.caption("🟢 ↑ lowers MMR · 🟠 ↑ raises MMR")

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
                        label = f"🟢 {col}" if is_protective_indicator(col) else f"🟠 {col}"
                        slider_key = f"lev_{country}_{col}_v{reset_ver}"
                        scenario_inputs[col] = st.slider(
                            label,
                            float(lo),
                            float(hi),
                            _slider_default(country, col, baseline),
                            float(step),
                            help=indicator_direction_hint(col),
                            key=slider_key,
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
                            default = _default_annual_delta(col)
                            step = 0.1
                            if col in ("Fertility Rate",):
                                step = 0.01
                            if col.endswith("(0/1)") or BOUNDS.get(col, (0, 2))[1] <= 1.0:
                                step = 0.01
                            delta_key = f"delta_{country}_{col}_v{reset_ver}"
                            annual_delta[col] = float(
                                st.number_input(
                                    f"{col} Δ/year",
                                    value=float(default),
                                    step=float(step),
                                    key=delta_key,
                                )
                            )

        with col_main:
            lever_changes = sum(
                abs(float(scenario_inputs.get(c, baseline[c])) - baseline[c]) for c in baseline
            )
            at_status_quo = lever_changes < 1e-6
            if at_status_quo:
                st.success("**Status quo active** — all levers match latest observed data.")
            else:
                st.warning(
                    "**Scenario active** — levers differ from status quo. "
                    "Use **Reset to status quo defaults** to return to baseline."
                )

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
                text=compare_df["MMR"].round(0).astype(int),
                title=f"{country}: maternal mortality ratio comparison",
                color_discrete_sequence=["#3b82f6", "#009E73"],
            )
            fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            fig.update_layout(showlegend=False, yaxis_title="MMR per 100,000 live births")
            st.plotly_chart(fig, use_container_width=True)

            if target_result_key in st.session_state:
                tr = st.session_state[target_result_key]
                changes = tr.changes_table()
                if changes:
                    st.subheader("Reverse scenario — required indicator shifts")
                    st.caption(
                        f"Values estimated to move from observed MMR **{tr.observed_mmr:.0f}** "
                        f"toward target **{tr.target_mmr:.0f}** "
                        f"(model achieved **{tr.achieved_mmr:.0f}**)."
                    )
                    rev_df = pd.DataFrame(changes)
                    st.dataframe(rev_df, use_container_width=True, hide_index=True)
                    rev_fig = px.bar(
                        rev_df.head(12),
                        x="Change",
                        y="Indicator",
                        orientation="h",
                        color="Change",
                        color_continuous_scale="RdYlGn",
                        title="Top required indicator changes for target MMR",
                        text=rev_df.head(12)["Change"].round(0).astype(int),
                    )
                    rev_fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
                    rev_fig.update_layout(coloraxis_showscale=False, showlegend=False)
                    st.plotly_chart(rev_fig, use_container_width=True)

            lever_cols = [c for c in FEATURE_COLUMNS if c in baseline]
            lever_df = pd.DataFrame(
                {
                    "Indicator": lever_cols,
                    "Baseline": [baseline[c] for c in lever_cols],
                    "Scenario": [model_scenario[c] for c in lever_cols],
                    "If you increase this lever": [
                        "MMR should ↓" if is_protective_indicator(c) else "MMR should ↑"
                        for c in lever_cols
                    ],
                }
            )
            lever_df["Change"] = lever_df["Scenario"] - lever_df["Baseline"]
            st.subheader("Input changes")
            st.dataframe(lever_df.round(2), use_container_width=True, hide_index=True)
            if lever_df["Change"].abs().max() > 0:
                moved = lever_df[lever_df["Change"].abs() > 1e-9].copy()
                if not moved.empty:
                    st.caption(
                        "Check direction: for **MMR should ↓** rows, a positive **Change** "
                        "(you moved the slider up) should lower predicted MMR; for **MMR should ↑** "
                        "rows, a positive **Change** should raise predicted MMR."
                    )

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
                mmr_labels = [f"{v:.0f}" for v in proj["mmr_mean"]]
                figp.add_trace(
                    go.Scatter(
                        x=proj["year"],
                        y=proj["mmr_mean"],
                        mode="lines+markers+text",
                        name="MMR (mean)",
                        text=mmr_labels,
                        textposition="top center",
                        textfont=dict(size=11),
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
            export_ready = st.session_state.get("status_quo_confirmed", True)
            st.download_button(
                "Download scenario (Excel)",
                data=buffer.getvalue(),
                file_name=f"mmr_scenario_{country.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                disabled=not export_ready,
                help="Confirm status quo baseline in the left panel to enable export.",
            )

    elif section == "Preset comparisons":
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
            text=preset_df["Predicted MMR"].round(0).astype(int),
        )
        fig2.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig2.add_hline(
            y=pobserved, line_dash="dash", line_color="gray", annotation_text=f"Observed {pobserved:.0f}"
        )
        st.plotly_chart(fig2, use_container_width=True)

    elif section == "Regional data":
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

    elif section == "Data & methods":
        st.subheader("Methods")
        st.markdown(methods_markdown())
        st.subheader("Key data sources (public APIs)")
        st.markdown(
            """
            | Indicator | Source |
            |-----------|--------|
            | Maternal mortality ratio | [World Bank SH.STA.MMRT](https://data.worldbank.org/indicator/SH.STA.MMRT) (UN MMEIG) |
            | ANC4 coverage | [SP.DYN.4ANTE.ZS](https://data.worldbank.org/indicator/SP.DYN.4ANTE.ZS) |
            | Government health expenditure (% GDP) | [SH.XPD.GHED.GD.ZS](https://data.worldbank.org/indicator/SH.XPD.GHED.GD.ZS) |
            | Skilled birth attendance | [SH.STA.BRTW.ZS](https://data.worldbank.org/indicator/SH.STA.BRTW.ZS) |
            | Contraception (modern) | [SP.DYN.CONM.ZS](https://data.worldbank.org/indicator/SP.DYN.CONM.ZS) |
            | WHO GHO (C-section, workforce) | [GHO OData API](https://ghoapi.azureedge.net/api) |
            | UNICEF MNCH | [UNICEF SDMX](https://sdmx.data.unicef.org/) |

            Country list: **WHO African Region** member states. Full technical notes: `docs/Technical_Methodology.md`.
            """
        )
        if METHODOLOGY_DOCX.exists():
            st.download_button(
                "Download methodology (Word)",
                data=METHODOLOGY_DOCX.read_bytes(),
                file_name="Technical_Methodology.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        metrics_file = ROOT / METRICS_PATH
        if metrics_file.exists():
            st.json(json.loads(metrics_file.read_text(encoding="utf-8")))
