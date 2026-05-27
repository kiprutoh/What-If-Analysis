from __future__ import annotations

import streamlit as st

from src.app_page import APP_SECTIONS, PRESET_PACKAGE_COUNT, render_app
from src.home_metrics import compute_home_metrics


def _home_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; max-width: 1200px; }
          .mmh-wrap { color: #f8fafc; }
          .mmh-hero {
            position: relative; overflow: hidden; border-radius: 24px;
            background: linear-gradient(90deg, #0f172a 0%, #1e293b 50%, #334155 100%);
            border: 1px solid #334155; padding: 2.5rem; margin-bottom: 2rem;
            box-shadow: 0 25px 50px rgba(0,0,0,.35);
          }
          .mmh-hero-glow {
            position: absolute; right: 0; top: 0; width: 35%; height: 100%;
            background: radial-gradient(circle at center, rgba(239,68,68,.25), transparent 70%);
            pointer-events: none;
          }
          .mmh-badge {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 8px 16px; border-radius: 999px;
            background: rgba(239,68,68,.15); border: 1px solid rgba(248,113,113,.35);
            font-size: 0.9rem; margin-bottom: 1.25rem;
          }
          .mmh-title { font-size: 2.75rem; font-weight: 800; line-height: 1.1; margin: 0 0 1rem 0; }
          .mmh-gradient-text {
            background: linear-gradient(90deg, #f87171, #fb923c);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
          }
          .mmh-sub { color: #cbd5e1; font-size: 1.05rem; line-height: 1.65; max-width: 42rem; }
          .mmh-metric {
            border-radius: 16px; background: rgba(255,255,255,.05);
            border: 1px solid rgba(255,255,255,.1); padding: 1.1rem;
            backdrop-filter: blur(8px); height: 100%;
          }
          .mmh-metric-label { font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.35rem; }
          .mmh-metric-value { font-size: 2rem; font-weight: 800; margin: 0.15rem 0; }
          .mmh-metric-unit { font-size: 0.75rem; color: #94a3b8; }
          .mmh-metric-trend { font-size: 0.85rem; color: #34d399; font-weight: 600; margin-top: 0.5rem; }
          .mmh-card {
            border-radius: 24px; background: #0f172a; border: 1px solid #1e293b;
            padding: 2rem; height: 100%; box-shadow: 0 20px 40px rgba(0,0,0,.25);
            transition: transform .2s ease;
          }
          .mmh-card:hover { transform: translateY(-2px); }
          .mmh-card-stat { font-size: 0.72rem; letter-spacing: .12em; text-transform: uppercase; color: #94a3b8; }
          .mmh-card-title { font-size: 1.45rem; font-weight: 700; margin: 0.75rem 0 1rem 0; }
          .mmh-card-desc { color: #cbd5e1; line-height: 1.6; min-height: 4.5rem; }
          .mmh-card-footer { margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid #1e293b; color: #94a3b8; font-size: 0.9rem; }
          .mmh-mini-bars {
            display: flex; align-items: flex-end; gap: 6px; height: 96px;
            padding: 12px; border-radius: 16px; background: rgba(30,41,59,.7); border: 1px solid #334155;
            margin: 1.25rem 0;
          }
          .mmh-bar { flex: 1; border-radius: 10px 10px 0 0; }
          .mmh-step {
            border-radius: 24px; background: #0f172a; border: 1px solid #1e293b;
            padding: 1.5rem; position: relative; overflow: hidden; height: 100%;
          }
          .mmh-step-num-bg {
            position: absolute; top: -10px; right: 8px; font-size: 5rem; font-weight: 900;
            color: rgba(255,255,255,.04); line-height: 1;
          }
          .mmh-step-badge {
            width: 48px; height: 48px; border-radius: 16px; background: #ef4444;
            display: flex; align-items: center; justify-content: center;
            font-weight: 800; margin-bottom: 1rem;
          }
          .mmh-cta {
            border-radius: 24px; background: linear-gradient(90deg, #ef4444, #f97316);
            padding: 2.5rem; text-align: center; margin-top: 3rem;
            box-shadow: 0 20px 40px rgba(239,68,68,.25);
          }
          .mmh-cta h2 { font-size: 1.85rem; font-weight: 800; margin-bottom: 0.75rem; }
          .mmh-cta p { opacity: 0.92; max-width: 48rem; margin: 0 auto 1.5rem auto; line-height: 1.6; }
          div[data-testid="stAppViewContainer"] { background: #020617; }
          section[data-testid="stSidebar"] { background: #0f172a; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _nav_to(section: str) -> None:
    st.session_state["_mode"] = "What-if analysis"
    st.session_state["_app_section"] = section
    st.rerun()


def maternal_mortality_homepage() -> None:
    """Render the WHO AFRO maternal mortality landing page (Streamlit equivalent of MaternalMortalityHomepage)."""
    metrics_data = compute_home_metrics()
    n_countries = metrics_data["n_countries"]
    year_range = f"{metrics_data['year_min']}–{metrics_data['year_max']}"

    cards = [
        {
            "title": "Country Scenarios",
            "icon": "🌍",
            "description": (
                "Adjust health system levers and instantly visualize projected maternal "
                "mortality outcomes across countries."
            ),
            "stat": f"{n_countries} Countries",
            "gradient": "linear-gradient(135deg,#ef4444,#ec4899)",
            "footer": "policy simulations",
            "section": "Country scenario",
            "bars": [40, 55, 80, 65, 95],
            "colors": ["#f87171", "#fb923c", "#facc15", "#34d399", "#22d3ee"],
        },
        {
            "title": "Preset Intervention Packages",
            "icon": "📦",
            "description": (
                "Compare bundled interventions such as SBA coverage, literacy, and "
                "fertility reduction side-by-side."
            ),
            "stat": f"{PRESET_PACKAGE_COUNT} Scenario Packages",
            "gradient": "linear-gradient(135deg,#f97316,#f59e0b)",
            "footer": "planning workshops",
            "section": "Preset comparisons",
            "bars": [35, 50, 70, 60, 85],
            "colors": ["#fb923c", "#fdba74", "#fcd34d", "#fde68a", "#fbbf24"],
        },
        {
            "title": "Regional Insights",
            "icon": "📊",
            "description": (
                "Explore trends, correlations, and distributions of maternal mortality "
                "across WHO AFRO countries."
            ),
            "stat": f"{year_range} Trends",
            "gradient": "linear-gradient(135deg,#10b981,#14b8a6)",
            "footer": "situational analysis",
            "section": "Regional data",
            "bars": [30, 45, 60, 75, 90],
            "colors": ["#34d399", "#2dd4bf", "#5eead4", "#6ee7b7", "#14b8a6"],
        },
    ]

    metrics = [
        {
            "label": "Maternal Mortality Ratio",
            "value": f"{metrics_data['mmr_median']:.0f}",
            "unit": "per 100,000 live births (regional median)",
            "trend": metrics_data["mmr_trend"],
        },
        {
            "label": "Skilled Birth Attendance",
            "value": f"{metrics_data['sba_avg']:.0f}%",
            "unit": "regional median",
            "trend": "↑ improving coverage",
        },
        {
            "label": "Female Literacy",
            "value": f"{metrics_data['literacy_avg']:.0f}%",
            "unit": "median literacy rate",
            "trend": "↑ linked to lower MMR",
        },
        {
            "label": "Rural Population",
            "value": f"{metrics_data['rural_avg']:.0f}%",
            "unit": "population distribution",
            "trend": "Higher rural burden",
        },
    ]

    st.markdown('<div class="mmh-wrap">', unsafe_allow_html=True)

    # Hero
    metric_cells = "".join(
        f"""
        <div class="mmh-metric">
          <div class="mmh-metric-label">{m['label']}</div>
          <div class="mmh-metric-value">{m['value']}</div>
          <div class="mmh-metric-unit">{m['unit']}</div>
          <div class="mmh-metric-trend">{m['trend']}</div>
        </div>
        """
        for m in metrics
    )

    st.markdown(
        f"""
        <div class="mmh-hero">
          <div class="mmh-hero-glow"></div>
          <div style="display:grid;grid-template-columns:1.2fr 1fr;gap:2rem;align-items:center;position:relative;z-index:2;">
            <div>
              <div class="mmh-badge"><span>🩺</span> WHO AFRO Maternal Health Intelligence Platform</div>
              <h1 class="mmh-title">
                Maternal Mortality<br/>
                <span class="mmh-gradient-text">Scenario Explorer</span>
              </h1>
              <p class="mmh-sub">
                Explore how improvements in skilled birth attendance, ANC4 coverage, government
                health expenditure, fertility, female literacy, and rural access can influence
                maternal mortality outcomes across WHO AFRO countries.
              </p>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">{metric_cells}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hc1, hc2 = st.columns(2)
    with hc1:
        if st.button("Start What-if Analysis", type="primary", use_container_width=True, key="hero_start"):
            _nav_to("Country scenario")
    with hc2:
        if st.button("Explore Regional Insights", use_container_width=True, key="hero_regional"):
            _nav_to("Regional data")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Cards
    c1, c2, c3 = st.columns(3)
    for col, card in zip([c1, c2, c3], cards):
        bars_html = "".join(
            f'<div class="mmh-bar" style="height:{h}%;background:{c};"></div>'
            for h, c in zip(card["bars"], card["colors"])
        )
        with col:
            st.markdown(
                f"""
                <div class="mmh-card">
                  <div style="width:64px;height:64px;border-radius:16px;background:{card['gradient']};
                    display:flex;align-items:center;justify-content:center;font-size:1.75rem;margin-bottom:1.25rem;">{card['icon']}</div>
                  <div class="mmh-card-stat">{card['stat']}</div>
                  <div class="mmh-card-title">{card['title']}</div>
                  <div class="mmh-card-desc">{card['description']}</div>
                  <div class="mmh-mini-bars">{bars_html}</div>
                  <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#94a3b8;">
                    <span>Baseline</span><span>Scenario Impact</span>
                  </div>
                  <div class="mmh-card-footer"><strong style="color:#fff;">Best for:</strong> {card['footer']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Open {card['title']}", key=f"card_{card['section']}", use_container_width=True):
                _nav_to(card["section"])

    st.markdown("<br/>", unsafe_allow_html=True)

    # Workflow
    steps = [
        ("1", "Select Country", "Choose a WHO AFRO country dataset"),
        ("2", "Review Status Quo", "Reset levers to latest observed baseline values"),
        ("3", "Adjust Levers", "Modify SBA, ANC4, gov spending, literacy & rural access"),
        ("4", "Export Results", "Download Excel reports & visual summaries"),
    ]
    s1, s2, s3, s4 = st.columns(4)
    for col, step in zip([s1, s2, s3, s4], steps):
        with col:
            st.markdown(
                f"""
                <div class="mmh-step">
                  <div class="mmh-step-num-bg">{step[0]}</div>
                  <div class="mmh-step-badge">{step[0]}</div>
                  <h4 style="font-size:1.15rem;font-weight:700;margin-bottom:0.5rem;">{step[1]}</h4>
                  <p style="color:#94a3b8;line-height:1.55;margin:0;">{step[2]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="mmh-cta">
          <h2>Transform Maternal Health Planning with Data</h2>
          <p>Use interactive forecasting, regional insights, and scenario modelling to support
          evidence-based maternal health policy decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Launch Scenario Explorer", type="primary", use_container_width=True, key="cta_launch"):
        _nav_to("Country scenario")

    st.markdown("</div>", unsafe_allow_html=True)


MaternalMortalityHomepage = maternal_mortality_homepage


def main() -> None:
    st.set_page_config(
        page_title="WHO AFRO Maternal Mortality What-if",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if st.session_state.get("_mode") == "What-if analysis":
        render_app()
        return

    _home_css()
    with st.sidebar:
        st.markdown("### Navigation")
        if st.button("Home", use_container_width=True):
            st.session_state["_mode"] = "Home"
            st.rerun()
        if st.button("What-if analysis", use_container_width=True, type="primary"):
            _nav_to("Country scenario")
        st.markdown("---")
        st.markdown("### Quick guide")
        st.markdown(
            "- Review **status quo** baseline\n"
            "- Adjust levers (🟢 lowers MMR when ↑)\n"
            "- Export Excel reports"
        )

    maternal_mortality_homepage()


if __name__ == "__main__":
    main()
