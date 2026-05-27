from __future__ import annotations

from datetime import date

import streamlit as st

from src.app_page import render_app


def _css() -> None:
    st.markdown(
        """
        <style>
          .app-hero {
            padding: 22px 26px;
            border-radius: 18px;
            background: radial-gradient(1200px circle at 10% 0%, rgba(0,158,115,.18), transparent 45%),
                        radial-gradient(900px circle at 85% 10%, rgba(59,130,246,.16), transparent 45%),
                        linear-gradient(180deg, rgba(255,255,255,.95), rgba(255,255,255,.86));
            border: 1px solid rgba(15, 23, 42, .08);
          }
          .app-hero h1 {
            font-size: 2.0rem;
            line-height: 1.15;
            margin: 0 0 6px 0;
          }
          .app-hero p {
            margin: 0;
            color: rgba(15, 23, 42, .78);
            font-size: 1.0rem;
          }
          .app-badge {
            display: inline-flex;
            gap: 8px;
            align-items: center;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(15, 23, 42, .10);
            background: rgba(255,255,255,.75);
            font-size: 0.85rem;
          }
          .app-card {
            border-radius: 16px;
            padding: 16px 16px 12px 16px;
            border: 1px solid rgba(15, 23, 42, .08);
            background: rgba(255,255,255,.7);
            height: 100%;
          }
          .app-card h3 { margin: 0 0 6px 0; font-size: 1.05rem; }
          .app-card p { margin: 0; color: rgba(15, 23, 42, .78); }
          .app-divider { margin: 12px 0 2px 0; opacity: .45; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="WHO AFRO Maternal Mortality What-if",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _css()

    with st.sidebar:
        st.markdown("### Navigation")
        mode = st.radio(
            "Choose a view",
            ["Home", "What-if analysis"],
            index=0 if st.session_state.get("_mode", "Home") == "Home" else 1,
            label_visibility="collapsed",
        )
        st.session_state["_mode"] = mode

        st.markdown("---")
        st.markdown("### Quick guide")
        st.markdown(
            "- Pick a country and adjust levers\n"
            "- Compare scenario vs **model baseline**\n"
            "- Export an Excel for reporting"
        )

    if st.session_state["_mode"] == "What-if analysis":
        render_app()
        return

    # Home
    st.markdown(
        f"""
        <div class="app-hero">
          <div class="app-badge">🌍 WHO AFRO • Maternal health • What-if scenarios</div>
          <h1>Maternal Mortality Scenario Explorer</h1>
          <p>
            A ready-to-deploy interface for exploring how changes in key levers
            (SBA, fertility, female literacy, rural population) relate to maternal mortality ratio (MMR).
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="app-card">
              <h3>Country scenarios</h3>
              <p>Pick a country, adjust levers, and instantly see predicted MMR vs baseline.</p>
              <hr class="app-divider"/>
              <p><b>Best for:</b> planning discussions & policy dialogues</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="app-card">
              <h3>Preset packages</h3>
              <p>Compare bundled interventions (e.g., SBA + literacy + fertility shift) side-by-side.</p>
              <hr class="app-divider"/>
              <p><b>Best for:</b> scenario workshops</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="app-card">
              <h3>Regional insights</h3>
              <p>Explore maps, relationships, and distributions across WHO AFRO countries and years.</p>
              <hr class="app-divider"/>
              <p><b>Best for:</b> situational analysis</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    colA, colB = st.columns([2, 1], gap="large")
    with colA:
        st.subheader("Start a what-if analysis")
        st.markdown(
            "Use the sidebar to switch to **What-if analysis**. "
            "All outputs can be exported to Excel for reporting."
        )
        if st.button("Open What-if analysis", type="primary", use_container_width=True):
            st.session_state["_mode"] = "What-if analysis"
            st.rerun()
    with colB:
        st.subheader("Notes")
        st.markdown(
            f"- **Updated**: {date.today().isoformat()}\n"
            "- Uses public World Bank indicators\n"
            "- Predictions are illustrative (non-causal)"
        )


if __name__ == "__main__":
    main()

