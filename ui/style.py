"""Visual theme for the Streamlit dashboard: palette, CSS and the Plotly template."""

from __future__ import annotations

NAVY = "#1B2A4A"
NAVY_DARK = "#121D36"
NAVY_SOFT = "#E6EAF2"
BLUE_MID = "#2E6F9E"
AMBER = "#E0A43A"
SLATE = "#7D8CB0"
INK = "#16181D"
MUTED = "#5C606A"
BORDER = "#E0DFDB"
UP = "#13795B"
DOWN = "#B3402A"

# Three blues stepping dark to light: distinct in greyscale and for colour-blind readers.
# Never chart in amber, UP or DOWN (red and green read as loss and gain).
SERIES = [NAVY, BLUE_MID, SLATE]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], button, input, textarea {{
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}}
h1, h2, h3, h4 {{ font-family: 'IBM Plex Serif', Georgia, serif !important; font-weight: 600; letter-spacing: -0.01em; }}
code, .lm-mono {{ font-family: 'IBM Plex Mono', ui-monospace, monospace; font-feature-settings: "tnum"; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 4.5rem; max-width: 1360px; }}

.lm-bar {{ background: {NAVY}; color: #fff; border-radius: 12px; padding: 14px 20px; display: flex;
  align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 1rem; }}
.lm-bar .lm-logo {{ width: 28px; height: 28px; border-radius: 7px; background: {AMBER}; display: grid; place-items: center; }}
.lm-bar .lm-name {{ font-family: 'IBM Plex Serif', Georgia, serif; font-weight: 600; font-size: 18px; }}
.lm-bar .lm-pill {{ margin-left: auto; font-size: 12px; color: #D9DEEA; border: 1px solid rgba(255,255,255,.22);
  border-radius: 999px; padding: 4px 10px; display: inline-flex; gap: 8px; align-items: center; }}
.lm-bar .lm-dot {{ width: 8px; height: 8px; border-radius: 50%; background: {AMBER}; }}

.lm-hero {{ text-align: center; padding: 2.5rem 0 1rem; }}
.lm-hero h1 {{ font-size: 2.3rem; margin: 0 0 .4rem; }}
.lm-hero p {{ color: {MUTED}; font-size: 1.05rem; margin: 0; }}

.lm-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: .5rem 0 1rem; }}
.lm-kpi {{ background: #fff; border: 1px solid {BORDER}; border-radius: 10px; padding: 14px 16px; display: flex; flex-direction: column; gap: 4px; }}
.lm-kpi .lbl {{ font-size: 12px; color: {MUTED}; }}
.lm-kpi .val {{ font-family: 'IBM Plex Mono', monospace; font-size: 19px; white-space: nowrap; font-weight: 500; color: {INK}; }}
.lm-kpi .up {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: {UP}; }}
.lm-kpi .down {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: {DOWN}; }}
.lm-kpi .src {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: {MUTED}; border-top: 1px solid #EBEAE6; padding-top: 6px; margin-top: 2px; }}

.lm-trust {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; background: #fff;
  border: 1px solid {BORDER}; border-radius: 10px; padding: 12px 16px; margin-bottom: 1rem; }}
.lm-trust div {{ display: flex; align-items: center; gap: 10px; }}
.lm-trust .ic {{ width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center; font-weight: 700; font-size: 13px; flex: none; }}
.lm-trust .ok {{ background: #DFF0E7; color: #106B4F; }}
.lm-trust .bad {{ background: #FDECEA; color: #A8211A; }}
.lm-trust .warn {{ background: #FFF1D6; color: #8A5300; }}
.lm-trust b {{ font-family: 'IBM Plex Mono', monospace; font-weight: 500; display: block; }}
.lm-trust small {{ color: {MUTED}; font-size: 12px; }}

.lm-chip {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; margin: 0 6px 6px 0; }}
.lm-chip.co {{ background: {NAVY_SOFT}; color: {NAVY}; }}
.lm-chip.cmp {{ background: #DFF0E7; color: #106B4F; }}
.lm-chip.muted {{ background: #EFEEEA; color: {MUTED}; font-weight: 500; }}

.lm-source {{ background: #fff; border: 1px solid #EBCB8B; border-radius: 10px; padding: 14px 16px; }}
.lm-source mark {{ background: #FBE3A8; color: {INK}; padding: 1px 2px; border-radius: 2px; }}
.lm-removed {{ background: #FFF6F5; border: 1px solid #F2C9C5; border-radius: 10px; padding: 12px 16px; color: #6E1A15; font-size: 13px; margin-top: 12px; }}
.lm-removed b {{ color: #8C1D17; display: block; margin-bottom: 4px; }}

.lm-step {{ display: flex; gap: 10px; align-items: baseline; padding: 4px 0; font-size: 13px; }}
.lm-step .mk {{ width: 18px; height: 18px; border-radius: 50%; display: inline-grid; place-items: center; font-size: 11px; font-weight: 700; flex: none; }}
.lm-step code {{ background: none; color: {INK}; padding: 0; }}
.lm-step .dur {{ margin-left: auto; color: {MUTED}; font-family: 'IBM Plex Mono', monospace; font-size: 12px; }}

@keyframes lm-shimmer {{ 0% {{ background-position: -400px 0; }} 100% {{ background-position: 400px 0; }} }}
.lm-sk {{ height: 14px; border-radius: 6px; margin: 10px 0; background: linear-gradient(90deg,#EBEAE6 0,#F3F3F1 50%,#EBEAE6 100%);
  background-size: 800px 100%; animation: lm-shimmer 1.4s linear infinite; }}
@media (prefers-reduced-motion: reduce) {{ .lm-sk {{ animation: none; }} }}

.lm-foot {{ text-align: center; color: {MUTED}; font-size: 12px; margin-top: 2rem; }}
</style>
"""

PLOTLY_LAYOUT = dict(
    font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=13),
    colorway=SERIES,
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
    title=dict(font=dict(family="IBM Plex Serif, Georgia, serif", size=16)),
)
