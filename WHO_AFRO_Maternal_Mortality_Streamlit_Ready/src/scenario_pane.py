"""Left-pane styling for the country scenario workspace."""

from __future__ import annotations

import streamlit as st


def inject_left_pane_css() -> None:
    st.markdown(
        """
        <style>
          div[data-testid="column"]:has(.scenario-pane-anchor) {
            background: linear-gradient(165deg, #0f172a 0%, #1e293b 48%, #0f172a 100%);
            border-radius: 22px;
            border: 1px solid rgba(148, 163, 184, 0.22);
            padding: 0.35rem 0.85rem 1rem 0.85rem;
            box-shadow: 0 18px 40px rgba(2, 6, 23, 0.35);
          }
          div[data-testid="column"]:has(.scenario-pane-anchor) label,
          div[data-testid="column"]:has(.scenario-pane-anchor) .stMarkdown p,
          div[data-testid="column"]:has(.scenario-pane-anchor) .stMarkdown li {
            color: #e2e8f0 !important;
          }
          div[data-testid="column"]:has(.scenario-pane-anchor) .stCaption {
            color: #94a3b8 !important;
          }
          .scenario-pane-anchor { height: 0; overflow: hidden; }
          .scenario-pane-title {
            font-size: 1.75rem;
            font-weight: 800;
            line-height: 1.12;
            margin: 0.25rem 0 0.15rem 0;
            letter-spacing: -0.02em;
          }
          .scenario-pane-title .line1 { color: #f8fafc; display: block; }
          .scenario-pane-title .line2 {
            display: block;
            background: linear-gradient(90deg, #f87171, #fb923c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
          }
          .scenario-pane-sub {
            color: #94a3b8;
            font-size: 0.88rem;
            margin-bottom: 1rem;
            line-height: 1.5;
          }
          .scenario-pane-chip {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(248, 113, 113, 0.35);
            color: #fecaca;
            margin-bottom: 0.75rem;
          }
          .target-panel {
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 16px;
            padding: 0.85rem 1rem;
            margin: 0.75rem 0 1rem 0;
          }
          .target-panel h4 {
            color: #7dd3fc !important;
            margin: 0 0 0.35rem 0 !important;
            font-size: 1rem !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_left_pane_header() -> None:
    st.markdown('<div class="scenario-pane-anchor"></div>', unsafe_allow_html=True)
    st.markdown('<div class="scenario-pane-chip">WHO AFRO · What-if workspace</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="scenario-pane-title">
          <span class="line1">Maternal Mortality</span>
          <span class="line2">Scenario Explorer</span>
        </div>
        <p class="scenario-pane-sub">
          Set a target MMR to work backwards, or adjust levers forward from status quo.
        </p>
        """,
        unsafe_allow_html=True,
    )
