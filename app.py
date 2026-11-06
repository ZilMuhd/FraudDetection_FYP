"""
Graph-Based Financial Fraud Detection: results dashboard (Streamlit).

    streamlit run app.py

Reads dashboard_data/ (made by dashboard_prepare.py and dashboard_export_graph.py)
and the reported results in dashboard_results.py. No Neo4j needed while it runs.
Theme: .streamlit/config.toml defines light and dark; viewers switch in the menu (top right).
Layout and styling are custom HTML/CSS on top of Streamlit (see CSS below).
"""

import os
import re
import json
import html as _html
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import streamlit.components.v1 as components
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, roc_auc_score
import dashboard_results as R

DATA = "dashboard_data"
BLUE, ORANGE, AQUA, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#9a9890"
LEVELS = ["clean", "low", "medium", "high"]
PAGES = ["Introduction", "PaySim", "Simulation (best case)", "Noisy simulation"]

# Paste the user acceptance test form link here if UAT is required ("" hides the section).
FEEDBACK_URL = ""

st.set_page_config(page_title="Graph-Based Fraud Detection", layout="wide", initial_sidebar_state="collapsed")


def is_dark():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


DARK = is_dark()
T = ({"bg": "#0e1117", "card": "#171c24", "ink": "#eef1f5", "ink2": "#b7bdc8", "line": "#2b323d", "soft": "#1e242d",
      "track": "#29303b", "deep": "#1f5aa6", "insight": "#1f5aa6", "edge": "#56606e"}
     if DARK else
     {"bg": "#eceef2", "card": "#ffffff", "ink": "#12161d", "ink2": "#465062", "line": "#dfe3ea", "soft": "#f4f5f8",
      "track": "#e8ebf0", "deep": "#1d4f91", "insight": "#12161d", "edge": "#c3c8d1"})

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
:root {--bg:%(bg)s; --card:%(card)s; --ink:%(ink)s; --ink2:%(ink2)s; --line:%(line)s; --soft:%(soft)s; --track:%(track)s;
       --deep:%(deep)s; --insight:%(insight)s; --blue:#2a78d6; --orange:#eb6834;}
html, body, .stApp, .stMarkdown, button, input, textarea, [data-testid="stWidgetLabel"] {font-family: 'Manrope', system-ui, sans-serif !important;}
.stApp {background: var(--bg);}
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none;}
header[data-testid="stHeader"] {background: transparent; pointer-events: none;}
header[data-testid="stHeader"] * {pointer-events: none;}
header[data-testid="stHeader"] button, header[data-testid="stHeader"] a, [data-testid="stMainMenu"], [data-testid="stMainMenu"] * {pointer-events: auto;}
.block-container {padding-top: 0.6rem; max-width: 1280px;}
.stMarkdown p {color: var(--ink);}

/* sticky top navigation */
[data-testid="stLayoutWrapper"]:has(> .st-key-topnav), div:has(> .st-key-topnav):not([data-testid="stVerticalBlock"]) {
    position: sticky; top: 0; z-index: 990; background: var(--bg);}
.st-key-topnav {position: sticky; top: 0; z-index: 990; background: var(--bg); padding: 0.75rem 5rem 0.75rem 0;
                border-bottom: 1px solid var(--line); margin-bottom: 0.6rem;}
.brand {white-space: nowrap; display: flex; align-items: center; gap: 0.6rem; font-weight: 800; font-size: 1.02rem; color: var(--ink); line-height: 1.15;}
.brand small {display: block; font-weight: 500; font-size: 0.78rem; color: var(--ink2);}

/* pills: segmented controls and tabs */
[data-testid="stButtonGroup"] {gap: 0.35rem; flex-wrap: wrap;}
[data-testid="stButtonGroup"] button {border-radius: 999px !important; border: 1px solid var(--line) !important;
    background: var(--card) !important; color: var(--ink) !important; padding: 0.3rem 0.95rem !important; min-height: 2.1rem;}
[data-testid="stButtonGroup"] button p {font-weight: 600; font-size: 0.9rem;}
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stBaseButton-segmented_controlActive"] {background: var(--ink) !important; border-color: var(--ink) !important;}
[data-testid="stButtonGroup"] button[aria-checked="true"] p, [data-testid="stBaseButton-segmented_controlActive"] p {color: var(--bg) !important;}
.st-key-sections button[aria-checked="true"], .st-key-sections [data-testid="stBaseButton-segmented_controlActive"] {background: var(--deep) !important; border-color: var(--deep) !important;}
.st-key-sections button[aria-checked="true"] p, .st-key-sections [data-testid="stBaseButton-segmented_controlActive"] p {color: #fff !important;}
.stTabs [data-baseweb="tab-list"] {gap: 0.4rem; border: none;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}
.stTabs button[data-baseweb="tab"] {border-radius: 999px; padding: 0.4rem 1.1rem; background: transparent; height: auto;}
.stTabs button[data-baseweb="tab"] p {font-weight: 700; font-size: 0.95rem; color: var(--ink2);}
.stTabs button[aria-selected="true"] {background: var(--card);}
.stTabs button[aria-selected="true"] p {color: var(--ink);}

/* cards that hold widgets */
[class*="st-key-card_"] {background: var(--card); border-radius: 22px; padding: 1.2rem 1.4rem 1.1rem 1.4rem;}
[data-baseweb="select"] > div, [data-testid="stSelectbox"] .react-aria-ComboBox > div {background: var(--card) !important; border: 1.5px solid %(edge)s !important; border-radius: 12px !important;}
[data-testid="stExpander"] details {border: none !important; background: var(--card); border-radius: 18px;}

/* type */
.hero h1 {font-size: 2.7rem; font-weight: 800; letter-spacing: -0.035em; line-height: 1.04; margin: 0.6rem 0 0.6rem 0; color: var(--ink);}
.hero p {font-size: 1.06rem; color: var(--ink2); max-width: 62ch; line-height: 1.6; margin: 0;}
.sh {margin: 2rem 0 0.8rem 0;}
.sh h2 {font-size: 1.45rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 0.25rem 0; padding: 0; color: var(--ink);}
.sh p {color: var(--ink2); font-size: 0.99rem; max-width: 78ch; line-height: 1.55; margin: 0;}
.term {border-bottom: 1.5px dotted currentColor; cursor: help;}

/* grids and KPI cards */
.grid {display: grid; gap: 14px; grid-template-columns: repeat(var(--c), minmax(0, 1fr)); margin: 0.2rem 0 0.4rem 0;}
.kpi {background: var(--card); border-radius: 20px; padding: 1.05rem 1.2rem; color: var(--ink);}
.kpi .kl {font-size: 0.88rem; font-weight: 600; color: var(--ink2);}
.kpi .kv {font-size: 2.15rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.15; margin: 0.25rem 0 0.15rem 0;}
.kpi .kv span {font-size: 1rem; font-weight: 600; letter-spacing: 0; margin-left: 0.25rem; color: var(--ink2);}
.kpi .kn {font-size: 0.86rem; color: var(--ink2); line-height: 1.45;}
.kpi.sm .kv {font-size: 1.55rem;}
.kpi.feature {background: var(--deep); color: #fff;}
.kpi.feature .kl, .kpi.feature .kn, .kpi.feature .kv span {color: rgba(255,255,255,0.86);}
.box {background: var(--card); border-radius: 20px; padding: 1.15rem 1.3rem; color: var(--ink); display: flex; flex-direction: column;}
.box .step {align-self: flex-start;}
.box h4 {margin: 0 0 0.35rem 0; font-size: 1.02rem; font-weight: 800; padding: 0;}
.box p {margin: 0.25rem 0; font-size: 0.95rem; line-height: 1.55; color: var(--ink2);}
.box .big {font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; color: var(--blue); margin-top: auto; padding-top: 0.6rem;}
.step {display: inline-flex; width: 1.75rem; height: 1.75rem; border-radius: 50%; background: var(--ink); color: var(--bg);
       font-weight: 800; font-size: 0.9rem; align-items: center; justify-content: center; margin-bottom: 0.55rem;}
.tag {display: inline-block; font-size: 0.78rem; font-weight: 700; padding: 0.15rem 0.6rem; border-radius: 999px;
      background: rgba(255,255,255,0.18); color: #fff; margin-bottom: 0.5rem;}
.tag.warn {background: rgba(235,104,52,0.15); color: var(--orange);}

/* insight and note */
.insight {background: var(--insight); color: #fff; border-radius: 22px; padding: 1.2rem 1.5rem; margin: 1rem 0 0.4rem 0;}
.insight h4 {color: #fff; margin: 0 0 0.45rem 0; font-size: 1.02rem; font-weight: 800; padding: 0;}
.insight ul {margin: 0; padding-left: 1.15rem;} .insight li, .insight p {color: #fff; font-size: 0.98rem; line-height: 1.55; margin: 0.18rem 0;}
.note {background: var(--soft); border: 1px solid var(--line); border-radius: 16px; padding: 0.8rem 1.1rem; color: var(--ink2);
       font-size: 0.92rem; line-height: 1.55; margin: 0.7rem 0 0.2rem 0;}
.note b {color: var(--ink);}

/* bar charts */
.bars {background: var(--card); border-radius: 20px; padding: 1.1rem 1.3rem; color: var(--ink);}
.bars .bt {font-size: 0.95rem; font-weight: 800; margin-bottom: 0.2rem;}
.bars .bs {font-size: 0.86rem; color: var(--ink2); margin-bottom: 0.7rem;}
.br {display: grid; grid-template-columns: minmax(140px, 34%) 1fr; gap: 0.9rem; align-items: center; padding: 0.42rem 0;}
.br + .br {border-top: 1px solid var(--line);}
.bl {font-size: 0.9rem; font-weight: 600; line-height: 1.3;}
.bl small {display: block; font-weight: 500; color: var(--ink2); font-size: 0.8rem;}
.bline {display: grid; grid-template-columns: 1fr 5.6rem; gap: 0.6rem; align-items: center; margin: 0.12rem 0;}
.track {position: relative; height: 12px; border-radius: 99px; background: var(--track);}
.fill {height: 100%%; border-radius: 99px;}
.rng {position: absolute; top: -4px; height: 20px; border-left: 2px solid var(--ink); border-right: 2px solid var(--ink);}
.rng::after {content: ""; position: absolute; left: 0; right: 0; top: 9px; border-top: 2px solid var(--ink);}
.bv {font-size: 0.88rem; font-weight: 700; text-align: right; white-space: nowrap;}
.legend {display: flex; flex-wrap: wrap; gap: 1rem; font-size: 0.85rem; color: var(--ink2); margin: 0.2rem 0 0.5rem 0;}
.legend i {display: inline-block; width: 0.75rem; height: 0.75rem; border-radius: 3px; margin-right: 0.35rem; vertical-align: -1px;}
.legend i.dot {border-radius: 50%%;} .legend i.ln {height: 3px; width: 1.1rem; border-radius: 2px; vertical-align: 3px;}
.legend i.ring {border-radius: 50%%; background: transparent !important; border: 2.5px solid #6250d6;}
.legend i.dash {height: 0; width: 1.2rem; border-top: 3px dashed; border-radius: 0; background: none !important; vertical-align: 3px;}
.div {display: grid; grid-template-columns: minmax(150px, 30%) 1fr 1fr 4.2rem; align-items: center; gap: 0.4rem; padding: 0.3rem 0;}
.div .neg {display: flex; justify-content: flex-end;} .div .neg div, .div .pos div {height: 12px; border-radius: 99px;}
.div .mid {border-left: 1px solid var(--line);}

/* tables */
.tbl {width: 100%%; border-collapse: collapse; background: var(--card); border-radius: 18px; overflow: hidden; font-size: 0.92rem; color: var(--ink);}
.tbl th {text-align: left; font-weight: 700; color: var(--ink2); padding: 0.7rem 0.9rem; border-bottom: 1px solid var(--line); font-size: 0.85rem;}
.tbl td {padding: 0.62rem 0.9rem; border-bottom: 1px solid var(--line); vertical-align: top; line-height: 1.45;}
.tbl tr:last-child td {border-bottom: none;} .tbl .n {text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums;}

/* timelines */
.rw {display: flex; border-radius: 16px; overflow: hidden; margin: 0.2rem 0 0.3rem 0;}
.rw div {padding: 0.8rem 0.9rem; font-size: 0.86rem; line-height: 1.4; color: #fff;}
.rw b {display: block; font-size: 0.95rem;}
.tl {background: var(--card); border-radius: 20px; padding: 0.4rem 1.3rem;}
.tl .it {display: grid; grid-template-columns: 2.2rem 1fr 6rem; gap: 0.8rem; padding: 0.9rem 0; align-items: start; color: var(--ink);}
.tl .it + .it {border-top: 1px solid var(--line);}
.tl .it b {font-size: 0.98rem;} .tl .it p {margin: 0.15rem 0 0 0; color: var(--ink2); font-size: 0.92rem; line-height: 1.5;}
.tl .sc {text-align: right; font-weight: 800; font-size: 1.15rem;} .tl .sc small {display: block; font-size: 0.75rem; color: var(--ink2); font-weight: 600;}
.shapes {display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px;}
.shape {background: var(--card); border-radius: 18px; padding: 0.8rem 0.9rem; color: var(--ink);}
.shape b {font-size: 0.95rem;} .shape p {font-size: 0.85rem !important; margin: 0.2rem 0 0 0; color: var(--ink2); line-height: 1.45;}
.terms {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.4rem 1.6rem; padding: 0.3rem 0.2rem;}
.terms div {font-size: 0.9rem; line-height: 1.5; color: var(--ink2); padding: 0.4rem 0; border-bottom: 1px solid var(--line);}
.terms b {color: var(--ink);}
.art {background: var(--card); border-radius: 26px; padding: 0.6rem 0.8rem 0.9rem 0.8rem;}
.art p {font-size: 0.85rem; color: var(--ink2); margin: 0 0.6rem;}
@media (max-width: 900px) {.grid {grid-template-columns: repeat(2, minmax(0, 1fr));} .shapes, .terms {grid-template-columns: 1fr 1fr;}
  .st-key-topnav {padding-right: 0;} .hero h1 {font-size: 2rem;}}
@media (max-width: 560px) {.grid, .shapes, .terms {grid-template-columns: 1fr;}}
"""
CSS = CSS.replace("%%", "%")
for _k, _v in T.items():
    CSS = CSS.replace(f"%({_k})s", _v)
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- loading
def path(name):
    return os.path.join(DATA, name)


@st.cache_data(show_spinner=False)
def load_csv(name):
    return pd.read_csv(path(name)) if os.path.exists(path(name)) else None


@st.cache_data(show_spinner=False)
def load_json(name):
    if not os.path.exists(path(name)):
        return None
    with open(path(name)) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model(name):
    if not os.path.exists(path(name)):
        return None
    m = xgb.XGBClassifier()
    m.load_model(path(name))
    return m


def missing(what, how):
    st.info(f"{what} is not prepared yet. Run: `{how}`")


# ---------------------------------------------------------------- glossary
GLOSSARY = {
    "PR-AUC": "Main score from 0 to 1 for how well the model ranks fraud above normal transactions. Used instead of accuracy because fraud is rare. Random guessing scores the fraud rate.",
    "chance level": "The PR-AUC of random guessing, equal to the share of transactions that are fraud.",
    "ROC-AUC": "A second ranking score (0.5 = random, 1 = perfect). It looks good even on rare fraud, so PR-AUC is the main measure.",
    "precision": "Of the transactions flagged, the share that really are fraud.",
    "recall": "Of all fraud, the share the model catches.",
    "false alarms": "Normal transactions flagged as fraud. Each one costs an investigator's time.",
    "threshold": "The fraud score above which a transaction is flagged. Lower catches more fraud but raises more false alarms.",
    "rolling window": "Train on earlier steps, test on later steps the model never saw, with graph features taken only from a graph of earlier transactions.",
    "step": "One hour of simulated time. PaySim covers steps 1-743 (about a month).",
    "Louvain": "An algorithm that splits the network into communities: groups of accounts that mostly transact with each other.",
    "community size": "Number of accounts in an account's Louvain community.",
    "fraud density": "Share of transactions inside a community that are fraud.",
    "fraud isolated": "Share of all fraud inside communities that are at least 50% fraud. High means Louvain separates the rings.",
    "PageRank": "How central an account is in the money flow: high when it receives money from many accounts that are themselves central.",
    "Degree": "Number of transactions an account took part in, in the graph of earlier steps.",
    "mule": "An account used to receive and pass on stolen money.",
    "camouflage": "Fraud accounts also behaving like normal customers (shared victims, everyday payments), which hides them in the network.",
    "label noise": "Real fraud that was never reported, so it is labelled normal in the training data.",
    "reload": "Loading the same data into Neo4j again and re-running Louvain. With noise, the result changes between reloads.",
    "SHAP": "How much one piece of information pushed one transaction's fraud score up or down, compared with an average transaction.",
    "std": "Standard deviation: how much a result moved across 5 repeated runs. Small means stable.",
    "leak": "Information the model would not have in real use, which makes its score look better than it is.",
    "train-serve skew": "Features built one way for training and another way for prediction, so the model learns patterns that do not hold later.",
}


def t(word, key=None):
    """A term with a hover definition."""
    return f"<span class='term' title='{_html.escape(GLOSSARY[key or word], quote=True)}'>{word}</span>"


# ---------------------------------------------------------------- HTML components
def H(s):
    st.markdown(s, unsafe_allow_html=True)


def hero(title, desc):
    H(f"<div class='hero'><h1>{title}</h1><p>{desc}</p></div>")


def head(title, desc=None):
    H(f"<div class='sh'><h2>{title}</h2>" + (f"<p>{desc}</p>" if desc else "") + "</div>")


def kpi(label, value, note="", tone="", unit=""):
    return (f"<div class='kpi {tone}'><div class='kl'>{label}</div><div class='kv'>{value}"
            + (f"<span>{unit}</span>" if unit else "") + "</div>" + (f"<div class='kn'>{note}</div>" if note else "") + "</div>")


def grid(items, cols=4):
    H(f"<div class='grid' style='--c:{cols}'>" + "".join(items) + "</div>")


def box(title, body, big=None, step=None, tag=None):
    return ("<div class='box'>" + (f"<div class='step'>{step}</div>" if step else "") + (tag or "") + f"<h4>{title}</h4>"
            + "".join(f"<p>{p}</p>" for p in ([body] if isinstance(body, str) else body))
            + (f"<div class='big'>{big}</div>" if big else "") + "</div>")


def insight(points, title="What this shows"):
    body = f"<p>{points}</p>" if isinstance(points, str) else "<ul>" + "".join(f"<li>{p}</li>" for p in points) + "</ul>"
    H(f"<div class='insight'><h4>{title}</h4>{body}</div>")


def note(s):
    H(f"<div class='note'>{s}</div>")


def legend(items):
    """items: (color, text, kind) with kind in sq, dot, ln, ring, dash."""
    return "<div class='legend'>" + "".join(
        f"<span><i class='{k}' style='{'border-top-color' if k == 'dash' else 'background'}:{c}'></i>{x}</span>" for c, x, k in items) + "</div>"


def bars(rows, title=None, sub=None, vmax=None, leg=None):
    """rows: dicts with label, s = [(value, color, text)], optional lo/hi (range marker on the first series)."""
    vmax = vmax or max(v for r in rows for v, _, _ in r["s"]) or 1
    out = "<div class='bars'>" + (f"<div class='bt'>{title}</div>" if title else "") + (f"<div class='bs'>{sub}</div>" if sub else "")
    out += leg or ""
    for r in rows:
        out += f"<div class='br'><div class='bl'>{r['label']}</div><div>"
        for i, (v, c, txt) in enumerate(r["s"]):
            rng = ""
            if i == 0 and r.get("lo") is not None:
                rng = f"<span class='rng' style='left:{100 * r['lo'] / vmax:.1f}%;width:{max(0.6, 100 * (r['hi'] - r['lo']) / vmax):.1f}%'></span>"
            out += (f"<div class='bline'><div class='track'><div class='fill' style='width:{max(0.0, min(100, 100 * v / vmax)):.1f}%;"
                    f"background:{c}'></div>{rng}</div><div class='bv'>{txt}</div></div>")
        out += "</div></div>"
    H(out + "</div>")


def diverging(rows, title=None, sub=None):
    """rows: (label, contribution, value text). Orange to the right = towards fraud; blue to the left = towards normal."""
    m = max(abs(c) for _, c, _ in rows) or 1
    out = "<div class='bars'>" + (f"<div class='bt'>{title}</div>" if title else "") + (f"<div class='bs'>{sub}</div>" if sub else "")
    out += legend([(BLUE, "pushed towards normal", "sq"), (ORANGE, "pushed towards fraud", "sq")])
    for lab, c, val in rows:
        w = max(3.0, 100 * abs(c) / m) if c != 0 else 0
        neg = f"<div style='width:{w:.1f}%;background:{BLUE}'></div>" if c < 0 else ""
        pos = f"<div style='width:{w:.1f}%;background:{ORANGE}'></div>" if c >= 0 else ""
        out += (f"<div class='div'><div class='bl'>{lab}<small>value {val}</small></div><div class='neg'>{neg}</div>"
                f"<div class='pos mid'>{pos}</div><div class='bv'>{c:+.2f}</div></div>")
    H(out + "</div>")


def table(df, cols, num=(), fmt=None):
    """cols: {source column: header}. num: headers aligned right. fmt: {header: function}."""
    fmt = fmt or {}
    d = df.rename(columns=cols)[list(cols.values())]
    out = "<table class='tbl'><tr>" + "".join(f"<th class='{'n' if h in num else ''}'>{h}</th>" for h in d.columns) + "</tr>"
    for _, r in d.iterrows():
        cells = []
        for h in d.columns:
            v = r[h]
            s = fmt[h](v) if h in fmt else ("—" if isinstance(v, float) and np.isnan(v) else str(v))
            cells.append(f"<td class='{'n' if h in num else ''}'>{s}</td>")
        out += "<tr>" + "".join(cells) + "</tr>"
    H(out + "</table>")


f3 = lambda v: f"{v:.3f}"
f4 = lambda v: "—" if isinstance(v, float) and np.isnan(v) else f"{v:.4f}"
pc = lambda v: f"{v:.1%}"
p0 = lambda v: f"{v:.0%}"
ic = lambda v: f"{v:,.0f}"


def card(name):
    return st.container(key=f"card_{name}")


# ---------------------------------------------------------------- model helpers
def at_threshold(y, score, th):
    pred = score >= th
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return {"precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
            "caught": tp, "false_alarms": fp, "missed": fn}


def pr_curve_df(y, score, name, n=300):
    p, r, _ = precision_recall_curve(y, score)
    idx = np.unique(np.linspace(0, len(p) - 1, n).astype(int))
    return pd.DataFrame({"recall": r[idx], "precision": p[idx], "model": name})


def style(ch):
    return ch.configure(background="transparent", font="Manrope, system-ui, sans-serif").configure_view(strokeWidth=0).configure_axis(
        labelColor=T["ink2"], titleColor=T["ink2"], gridColor=T["line"], domainColor=T["line"], tickColor=T["line"],
        labelFontSize=12, titleFontSize=12, titleFontWeight=600).configure_legend(labelColor=T["ink"], labelFontSize=12)


def threshold_panel(y, scores, key, default=0.5, foot=""):
    """Live threshold: precision, recall and false alarms for each model at one cut-off, with the precision-recall curve."""
    with card(key):
        th = st.slider("Threshold: flag a transaction when its fraud score is at least", 0.01, 0.99, default, 0.01, key=key)
        rows, colors = [], [BLUE, ORANGE, AQUA]
        for name, s in scores.items():
            rows.append({"model": name, **at_threshold(y, s, th)})
        H("<div class='grid' style='--c:%d'>" % len(rows) + "".join(
            f"<div class='kpi sm' style='background:var(--soft)'><div class='kl'><i style='display:inline-block;width:.7rem;height:.7rem;"
            f"border-radius:50%;background:{c};margin-right:.4rem'></i>{r['model']}</div>"
            f"<div style='display:flex;gap:1.4rem;margin-top:.35rem'>"
            f"<div><div class='kv' style='font-size:1.45rem'>{r['precision']:.0%}</div><div class='kn'>precision</div></div>"
            f"<div><div class='kv' style='font-size:1.45rem'>{r['recall']:.0%}</div><div class='kn'>recall</div></div>"
            f"<div><div class='kv' style='font-size:1.45rem'>{r['false_alarms']:,}</div><div class='kn'>false alarms</div></div></div>"
            f"<div class='kn' style='margin-top:.3rem'>Catches {r['caught']:,} of {r['caught'] + r['missed']:,} fraud transactions</div></div>"
            for r, c in zip(rows, colors)) + "</div>")
        curves = pd.concat([pr_curve_df(y, s, n) for n, s in scores.items()])
        pts = pd.DataFrame([{"model": r["model"], "recall": r["recall"], "precision": r["precision"]} for r in rows])
        names = list(scores)
        cs = alt.Scale(domain=names, range=colors[: len(names)])
        line = alt.Chart(curves).mark_line(strokeWidth=2.5).encode(
            x=alt.X("recall:Q", title="Recall (share of fraud caught)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("precision:Q", title="Precision (share of flags that are fraud)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model:N", scale=cs, legend=None),
            tooltip=["model", alt.Tooltip("recall:Q", format=".2f"), alt.Tooltip("precision:Q", format=".2f")])
        dot = alt.Chart(pts).mark_point(size=150, filled=True, stroke="white", strokeWidth=2).encode(
            x="recall:Q", y="precision:Q", color=alt.Color("model:N", scale=cs, legend=None),
            tooltip=["model", alt.Tooltip("recall:Q", format=".1%"), alt.Tooltip("precision:Q", format=".1%")])
        st.altair_chart(style((line + dot).properties(height=300)), use_container_width=True)
    note("<b>How to read it.</b> Lowering the threshold flags more transactions: more fraud is caught (recall up) but more "
         "normal transactions are flagged too (precision down). Raising it does the opposite. There is no single correct value; "
         "a bank sets it by how many alerts its team can check. The curve shows every possible trade-off and the dot marks the "
         "current threshold. A curve that stays higher is the better model. " + foot)


def explain(model, row, feats):
    dm = xgb.DMatrix(row[feats].astype(float).values.reshape(1, -1), feature_names=feats)
    contrib = model.get_booster().predict(dm, pred_contribs=True)[0]
    return pd.DataFrame({"feature": feats, "value": row[feats].values, "contribution": contrib[:-1]}), float(contrib[-1])


def nice(v):
    v = float(v)
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.2f}" if abs(v) >= 100 else f"{v:.3f}"


FEATURE_TEXT = {
    "origin_pageRank": "Sender's PageRank", "origin_degree": "Sender's transactions",
    "origin_communitySize": "Sender's community size", "dest_pageRank": "Receiver's PageRank",
    "dest_degree": "Receiver's transactions", "dest_communitySize": "Receiver's community size",
    "amount": "Amount", "hour": "Hour of day", "step": "Time step", "type_is_transfer": "Type (TRANSFER = 1)",
}


def inspector(df, label_col, score_col, model, feats, about):
    head("Why did the model score this transaction this way?",
         f"Pick a transaction and see which pieces of information raised or lowered its fraud score, measured with {t('SHAP')}. "
         + about)
    groups_def = {  # rule, and whether "most confident" means lowest score first
        "Fraud caught": ((lambda s, y, th: (s >= th) & (y == 1)), False),
        "Fraud missed": ((lambda s, y, th: (s < th) & (y == 1)), True),
        "False alarm": ((lambda s, y, th: (s >= th) & (y == 0)), False),
        "Correctly cleared": ((lambda s, y, th: (s < th) & (y == 0)), True),
    }
    why = {"Fraud caught": "fraud the model flagged", "Fraud missed": "fraud the model let through",
           "False alarm": "normal transactions flagged as fraud", "Correctly cleared": "normal transactions the model let through"}
    c1, c2 = st.columns([1, 2.15], gap="medium")
    y, s = df[label_col].values, df[score_col].values
    with c1:
        with card(f"ins_{score_col}"):
            th = st.slider("Threshold for counting as flagged", 0.01, 0.99, 0.5, 0.01, key=f"insp_{score_col}")
            kind = st.radio("Group", list(groups_def), key=f"kind_{score_col}")
            order = st.radio("Order", ["Most confident first", "Borderline first"], key=f"ord_{score_col}", horizontal=True)
            mask = groups_def[kind][0](s, y, th)
            g = df[mask]
            if order == "Most confident first":
                g = g.sort_values(score_col, ascending=groups_def[kind][1], kind="stable")
            else:
                g = g.iloc[np.argsort(np.abs(g[score_col].values - th), kind="stable")]
            g = g.head(200)
            H(f"<div class='kn' style='color:var(--ink2);font-size:.88rem;margin:-.2rem 0 .6rem 0'>{int(mask.sum()):,} {why[kind]} "
              f"at this threshold. Listing the first 200, {order.lower()}.</div>")
            if g.empty:
                st.info("No transactions in this group at this threshold.")
                return
            tcol = "step" if "step" in g.columns else None
            pick = st.selectbox("Transaction", list(range(len(g))), key=f"pick_{score_col}_{kind}_{order}", format_func=lambda i: (
                f"#{i + 1}   score {g.iloc[i][score_col]:.3f}   amount {g.iloc[i]['amount']:,.0f}"
                + (f"   step {int(g.iloc[i][tcol])}" if tcol else "")))
        note("<b>What the threshold changes.</b> It moves transactions between groups, so the list and the transaction in "
             "position #1 can change. A given transaction's fraud score and bars never change: they come from the model alone. "
             "<b>Borderline first</b> lists the scores closest to the threshold, where the reasons are most mixed.")
    row = g.iloc[pick]
    with c2:
        grid([kpi("Fraud score", f"{row[score_col]:.3f}", "0 = surely normal, 1 = surely fraud", "sm"),
              kpi("Actually fraud?", "Yes" if row[label_col] == 1 else "No", "", "sm"),
              kpi("Flagged at this threshold?", "Yes" if row[score_col] >= th else "No", "", "sm")], 3)
        if "isFraud_true" in df.columns and row["isFraud_true"] == 1 and row["isFraud"] == 0:
            note(f"<b>Unreported fraud.</b> This is real fraud whose label was missing in the bank's records ({t('label noise')}). "
                 "The model never saw it labelled as fraud in training.")
        if model is None:
            st.info("Model file missing, so the explanation cannot be shown.")
            return
        ex, base = explain(model, row, feats)
        ex["label"] = ex["feature"].map(lambda f: FEATURE_TEXT.get(f, f))
        ex = ex.reindex(ex["contribution"].sort_values(ascending=False).index)
        diverging([(r.label, r.contribution, nice(r.value)) for r in ex.itertuples()], "What drove this score",
                  "Each row is one piece of information about this transaction. A longer bar had a bigger effect. "
                  "Numbers are in log-odds, the model's internal scale (explained below).")
        top = ex.reindex(ex["contribution"].abs().sort_values(ascending=False).index).head(3)
        up = [f"{r.label} ({nice(r.value)})" for r in top.itertuples() if r.contribution > 0]
        down = [f"{r.label} ({nice(r.value)})" for r in top.itertuples() if r.contribution <= 0]
        pts = ([f"Raised the score most: {', '.join(up)}."] if up else []) + ([f"Lowered the score most: {', '.join(down)}."] if down else [])
        insight(pts, "The strongest reasons, in words")
        total = base + float(ex["contribution"].sum())
        note(f"<b>How the numbers add up.</b> The model starts every transaction from the same point, the average case "
             f"({base:+.2f}). Adding all the bars gives {total:+.2f}, which converts to the fraud score "
             f"{1 / (1 + np.exp(-total)):.3f}. Each +1 roughly multiplies the odds of fraud by 2.7 and each -1 divides them by "
             "2.7, so +12 means the evidence is overwhelming. The numbers are not percentages; compare their sizes and signs.")
        note("<b>What the features mean.</b> PageRank: how central the account is in the money flow of the earlier graph. "
             "Transactions: how many transactions the account had in that graph (Degree). Community size: accounts in its Louvain "
             "community. Type: 1 for a transfer to another account, 0 for a cash withdrawal. New accounts with no history have all graph values at 0.")


def network(nodes, edges, centre=None, scored=None, height=540):
    """Interactive graph (PyVis, JavaScript embedded so it works offline)."""
    from pyvis.network import Network
    net = Network(height=f"{height}px", width="100%", directed=True, bgcolor=T["card"], font_color=T["ink"], cdn_resources="in_line")
    degs = [max(1, n.get("degree") or 1) for n in nodes]
    dmax = max(degs) if degs else 1
    sender = scored["sender"] if scored else None
    for n, d in zip(nodes, degs):
        is_centre = n["id"] == centre
        role = " (receiver being scored)" if is_centre else " (sender of the scored transaction)" if n["id"] == sender else ""
        tip = [f"Account {n['id']}{role}", f"Fraud in its history: {'yes' if n.get('fraud') else 'no'}",
               f"Transactions in history: {n.get('degree', 0):,}"]
        if n.get("pageRank") is not None:
            tip.append(f"PageRank: {n['pageRank']:.3f}")
        if n.get("communitySize") is not None:
            tip.append(f"Community size: {int(n['communitySize']):,}")
        net.add_node(n["id"], label=" ", title="\n".join(tip),
                     color={"background": ORANGE if n.get("fraud") else BLUE, "border": VIOLET if is_centre else T["card"]},
                     borderWidth=5 if is_centre else 1, size=(34 if is_centre else 6 + 22 * (d / dmax) ** 0.5))
    for e in edges:
        fraud = (e.get("fraud") or 0) > 0
        net.add_edge(e["source"], e["target"], color=ORANGE if fraud else T["edge"], width=2.5 if fraud else 1,
                     title=f"{e.get('count', 1)} transaction(s)" + (", fraud" if fraud else ""), arrowStrikethrough=False)
    if scored:
        net.add_edge(scored["sender"], centre, color=ORANGE if scored["fraud"] else T["ink"], width=5, dashes=True,
                     title=f"Transaction being scored (step {scored['step']}): {'FRAUD' if scored['fraud'] else 'normal'}, "
                           f"amount {scored['amount']:,.0f}", arrowStrikethrough=False)
    net.set_options("""{"physics": {"solver": "forceAtlas2Based", "stabilization": {"enabled": true, "iterations": 400, "fit": true},
        "forceAtlas2Based": {"gravitationalConstant": -40, "springLength": 60}},
        "layout": {"randomSeed": 7}, "interaction": {"hover": true, "tooltipDelay": 80}, "edges": {"arrows": {"to": {"scaleFactor": 0.4}}}}""")
    html = re.sub(r'<(link|script)[^>]+cdn\.jsdelivr\.net[^>]*>(\s*</script>)?', "", net.generate_html())
    html = html.replace("#mynetwork {", "#mynetwork { border: none !important;")
    html = html.replace(
        "return network;",
        "network.once('stabilizationIterationsDone', function () { network.setOptions({physics: false}); network.fit(); });"
        " setTimeout(function () { network.redraw(); network.fit(); }, 1200);"
        " setTimeout(function () { network.setOptions({physics: false}); network.fit(); }, 4000); return network;")
    if scored:
        items = [(ORANGE, "fraud in its history (steps 1-594)", "dot"), (BLUE, "no fraud in its history", "dot"),
                 (ORANGE, "earlier fraud transaction", "ln"), (T["edge"], "earlier normal transaction", "ln")]
    else:
        items = [(ORANGE, "account involved in fraud", "dot"), (BLUE, "normal account", "dot"),
                 (ORANGE, "fraud transaction", "ln"), (T["edge"], "normal transaction", "ln")]
    if centre:
        items.append((VIOLET, "receiver being scored", "ring"))
    if scored:
        items += [(ORANGE, "transaction being scored: fraud", "dash"), (T["ink"], "transaction being scored: normal", "dash")]
    with card(f"net_{centre or len(nodes)}"):
        H(legend(items) + "<div class='kn' style='color:var(--ink2);font-size:.85rem;margin-top:-.2rem'>Dot size = number of "
          "transactions. Hover over a dot for details, drag to move, scroll to zoom.</div>")
        components.html(html, height=height + 10, scrolling=False)


def shapes_html():
    def svg(nodes, links, mid):
        r = 9
        parts = [f"<svg viewBox='0 0 160 110' width='100%' height='92'><defs><marker id='a{mid}' viewBox='0 0 10 10' refX='9' refY='5' "
                 f"markerWidth='6' markerHeight='6' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='{GREY}'/></marker></defs>"]
        for a, b in links:
            (x1, y1), (x2, y2) = nodes[a], nodes[b]
            dx, dy = x2 - x1, y2 - y1
            L = (dx * dx + dy * dy) ** 0.5
            ux, uy = dx / L, dy / L
            parts.append(f"<line x1='{x1 + ux * r:.1f}' y1='{y1 + uy * r:.1f}' x2='{x2 - ux * (r + 2):.1f}' y2='{y2 - uy * (r + 2):.1f}' "
                         f"stroke='{GREY}' stroke-width='2' marker-end='url(#a{mid})'/>")
        parts += [f"<circle cx='{x}' cy='{y}' r='{r}' fill='{ORANGE}'/>" for x, y in nodes]
        return "".join(parts) + "</svg>"
    shapes = [
        ("Chain", "Money passes along a line of mules.", [(18, 55), (59, 55), (100, 55), (141, 55)], [(0, 1), (1, 2), (2, 3)]),
        ("Fan-in", "Many accounts send into one collector.", [(25, 18), (25, 55), (25, 92), (130, 55)], [(0, 3), (1, 3), (2, 3)]),
        ("Fan-out", "One account splits money out to many.", [(30, 55), (135, 18), (135, 55), (135, 92)], [(0, 1), (0, 2), (0, 3)]),
        ("Cycle", "Money goes round a loop back to the start.", [(80, 16), (132, 92), (28, 92)], [(0, 1), (1, 2), (2, 0)]),
        ("Scatter-gather", "Split across several, then gathered into one.",
         [(18, 55), (80, 18), (80, 55), (80, 92), (142, 55)], [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)]),
    ]
    return "<div class='shapes'>" + "".join(
        f"<div class='shape'><b>{n}</b>{svg(nd, lk, i)}<p>{x}</p></div>" for i, (n, x, nd, lk) in enumerate(shapes)) + "</div>"


def hero_art():
    """The Start page picture: a fraud ring (orange) sitting inside a network of normal accounts (blue)."""
    rng = np.random.default_rng(11)
    W, Hh = 560, 330
    ring_c = np.array([392, 150])
    ring = [ring_c + 46 * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 7, endpoint=False)] + [ring_c]
    blue = []
    while len(blue) < 46:
        p = rng.uniform([22, 22], [W - 22, Hh - 22])
        if np.linalg.norm(p - ring_c) > 95 and all(np.linalg.norm(p - q) > 34 for q in blue):
            blue.append(p)
    lines = []
    for i, p in enumerate(blue):
        d = sorted(range(len(blue)), key=lambda j: np.linalg.norm(blue[j] - p))
        for j in d[1:3]:
            lines.append((p, blue[j], T["edge"], 1.3))
    for i in range(len(ring)):
        for j in range(i + 1, len(ring)):
            if i == len(ring) - 1 or (j - i) in (1, 2, 6, 5):
                lines.append((ring[i], ring[j], ORANGE, 2.2))
    near = sorted(range(len(blue)), key=lambda j: np.linalg.norm(blue[j] - ring_c))[:3]
    for k, j in enumerate(near):
        lines.append((ring[k * 2], blue[j], ORANGE, 1.6))
    svg = f"<svg viewBox='0 0 {W} {Hh}' width='100%' role='img' aria-label='A fraud ring inside a network of normal accounts'>"
    svg += "".join(f"<line x1='{a[0]:.0f}' y1='{a[1]:.0f}' x2='{b[0]:.0f}' y2='{b[1]:.0f}' stroke='{c}' stroke-width='{w}' "
                   f"stroke-linecap='round' opacity='{0.95 if c == ORANGE else 0.8}'/>" for a, b, c, w in lines)
    svg += "".join(f"<circle cx='{p[0]:.0f}' cy='{p[1]:.0f}' r='{rng.uniform(4, 8):.1f}' fill='{BLUE}'/>" for p in blue)
    svg += "".join(f"<circle cx='{p[0]:.0f}' cy='{p[1]:.0f}' r='{9 if k < 7 else 12}' fill='{ORANGE}'/>" for k, p in enumerate(ring))
    svg += (f"<circle cx='{ring_c[0]}' cy='{ring_c[1]}' r='74' fill='none' stroke='{ORANGE}' stroke-width='1.5' stroke-dasharray='4 5'/>"
            f"<text x='{ring_c[0]}' y='{ring_c[1] - 84}' text-anchor='middle' font-size='13' font-weight='700' fill='{ORANGE}' "
            "font-family='Manrope, sans-serif'>a reused mule ring</text></svg>")
    return (f"<div class='art'>{svg}<p>Each dot is an account and each line a transaction. The orange accounts keep "
            "passing money between each other: the kind of pattern no single transaction reveals.</p></div>")


# ---------------------------------------------------------------- navigation
def _keep(key):
    if st.session_state.get(key) is None:
        st.session_state[key] = st.session_state.get(f"_last_{key}")


def pick(label, options, key, default, collapsed=True, where=st):
    if key not in st.session_state:
        st.session_state[key] = default
    v = where.segmented_control(label, options, key=key, on_change=_keep, args=(key,),
                                label_visibility="collapsed" if collapsed else "visible")
    if v is None:
        v = st.session_state.get(f"_last_{key}", default)
    st.session_state[f"_last_{key}"] = v
    return v


def goto(page):
    st.session_state["page"] = page


BRAND = ("<div class='brand'><svg width='30' height='30' viewBox='0 0 30 30'><line x1='7' y1='8' x2='22' y2='11' stroke='%s' stroke-width='2'/>"
         "<line x1='7' y1='8' x2='13' y2='23' stroke='%s' stroke-width='2'/><line x1='22' y1='11' x2='13' y2='23' stroke='%s' stroke-width='2'/>"
         "<circle cx='7' cy='8' r='4' fill='%s'/><circle cx='22' cy='11' r='4' fill='%s'/><circle cx='13' cy='23' r='4' fill='%s'/></svg>"
         "<div>Graph fraud detection<small>FYP2 results dashboard</small></div></div>") % (ORANGE, ORANGE, ORANGE, BLUE, ORANGE, ORANGE)

with st.container(key="topnav"):
    a, b = st.columns([1.45, 4.85], vertical_alignment="center")
    a.markdown(BRAND, unsafe_allow_html=True)
    with b:
        page = pick("Dataset", PAGES, "page", "Introduction")
    level = None
    if page == "Noisy simulation":
        a2, b2 = st.columns([1.45, 4.85], vertical_alignment="center")
        a2.markdown("<div style='font-weight:700;font-size:.92rem;color:var(--ink2);text-align:right;padding-right:.4rem'>"
                    "Noise level</div>", unsafe_allow_html=True)
        with b2:
            level = pick("Noise level", LEVELS, "level", "clean")

man = load_json("manifest.json")
def _half_up(o):
    """Reported values like 0.7755 are stored in binary as 0.77549999..., so Python would show 0.775 while the report
    rounds half up to 0.776. A tiny nudge makes every displayed rounding match the report."""
    if isinstance(o, float) and not np.isnan(o):
        return o + 1e-9 if o >= 0 else o - 1e-9
    if isinstance(o, dict):
        return {k: _half_up(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_half_up(v) for v in o]
    return o


P, S, N = (_half_up(x) for x in (R.PAYSIM, R.SIMULATION, R.NOISY))

# ================================================================ START HERE
if page == "Introduction":
    g, ng = P["rolling"]["models"]
    c1, c2 = st.columns([1.05, 1], gap="large", vertical_alignment="center")
    with c1:
        hero("Fraud hides in who pays whom",
             "Criminals move stolen money through networks of mule accounts. One transaction on its own often looks normal; "
             "the pattern of payments around it often does not. This project turns transactions into a network, measures where "
             "each account sits in it, and lets an XGBoost model use those measurements to score every transaction for fraud "
             "and explain its decision.")
        H("<p style='color:var(--ink2);font-size:.95rem;margin-top:1rem'>Final Year Project: Graph-Based Financial Fraud Detection "
          "System Using Community Detection and Predictive Modelling.</p>")
    with c2:
        H(hero_art())

    head("How the system works", "Four stages, run in order for every dataset.")
    grid([box("Build the network", "Accounts become points and transactions become links, stored in a Neo4j graph database.", step=1),
          box("Find groups", f"{t('Louvain')} splits the network into communities. A fraud ring should form a group of its own.", step=2),
          box("Measure each account", f"{t('PageRank')} (how central it is), {t('Degree')} (how many transactions) and community size.", step=3),
          box("Predict and explain", f"XGBoost scores each transaction from 0 to 1, and {t('SHAP')} shows what drove each score.", step=4)], 4)

    head("Three datasets, three questions", "Each dataset tests a different part of the claim. Open one to see the evidence.")
    grid([box("PaySim: does it work on realistic data?",
              "2.77 million synthetic mobile-money transactions, tested on future data the model never saw. "
              "Fewer false alarms with graph features than without:", big=f"{1 - g['false_alarms'] / ng['false_alarms']:.0%} fewer"),
          box("Simulation: does Louvain find fraud rings?",
              "Forty fraud rings that reuse their mule accounts, the case PaySim lacks. Fraud share inside every top ring Louvain found:",
              big="100%"),
          box("Noisy simulation: what if fraud hides?",
              f"The same rings made harder to see, step by step, using published fraud behaviour. Final model's {t('PR-AUC')} at the "
              f"highest noise, against {N['levels']['high']['chance']:.3f} for random guessing:", big=f"{N['levels']['high']['G-noCS']:.2f}")], 3)
    b1, b2, b3 = st.columns(3)
    b1.button("Open PaySim", on_click=goto, args=("PaySim",), use_container_width=True)
    b2.button("Open the simulation", on_click=goto, args=("Simulation (best case)",), use_container_width=True)
    b3.button("Open the noisy simulation", on_click=goto, args=("Noisy simulation",), use_container_width=True)

    head("Project objectives", "What each objective set out to do, what was found, and where to see it.")
    where = ["Simulation and Noisy simulation: Network and Results", "Every dataset: Results and Inspector", "PaySim: Results, Future data and Threshold"]
    short = ["Simulation: fraud rings 100% fraud, normal communities 0%. PaySim: no rings, because it never reuses mule accounts. "
             "Noisy simulation: camouflage cuts isolation from about 87% to 7-18%, and Louvain's result varies between reloads of the same data.",
             "Built end to end. Graph features are among the top contributors in every final model.",
             "PaySim future data: 62% fewer false alarms and 26% more fraud caught than without graph features, across 3 algorithms, 5 seeds and two boundaries."]
    grid([box(o["objective"], [s_, f"<b style='color:var(--ink)'>Where to look:</b> {w}"]) for o, s_, w in zip(R.OBJECTIVES, short, where)], 3)

    head("Words used in this dashboard", "Underlined words elsewhere show the same definition when you hover over them.")
    keys = ["PR-AUC", "chance level", "precision", "recall", "false alarms", "threshold", "rolling window", "step", "Louvain",
            "community size", "fraud density", "fraud isolated", "PageRank", "Degree", "mule", "camouflage", "label noise", "SHAP"]
    with st.expander("Show all definitions"):
        H("<div class='terms'>" + "".join(f"<div><b>{k[0].upper() + k[1:]}</b>: {GLOSSARY[k]}</div>" for k in keys) + "</div>")
    if FEEDBACK_URL:
        head("Help improve this dashboard", "After exploring, please share your feedback in a short form.")
        st.link_button("Open the feedback form", FEEDBACK_URL)
    st.stop()

tabs = st.tabs(["Overview", "Network", "Results", "Inspector"])

# ================================================================ PAYSIM
if page == "PaySim":
    rw = load_csv("paysim_rw_test.csv.gz")
    g, ng = P["rolling"]["models"]
    fa_cut, caught_up = 1 - g["false_alarms"] / ng["false_alarms"], g["caught"] / ng["caught"] - 1
    with tabs[0]:
        hero("PaySim", "2.77 million synthetic mobile-money transactions. The model learns from earlier weeks and is tested on the "
             f"final weeks, which it never saw: the {t('rolling window', 'rolling window')} test.")
        H("<div style='height:.9rem'></div>")
        grid([kpi(f"{t('PR-AUC')} on future data", f"{g['pr_auc']:.3f}", f"Without graph features: {ng['pr_auc']:.3f}", "feature"),
              kpi(t("False alarms", "false alarms"), f"{g['false_alarms']:,}", f"{fa_cut:.0%} fewer than without graph features ({ng['false_alarms']:,})"),
              kpi("Fraud caught", f"{g['caught']:,}", f"{caught_up:.0%} more than without graph features ({ng['caught']:,})"),
              kpi(t("Chance level", "chance level"), f"{P['rolling']['chance']:.3f}", "What random guessing would score")], 4)
        head("The data", "Only TRANSFER and CASH_OUT transactions, the two types where PaySim fraud occurs.")
        grid([kpi("Transactions", f"{P['dataset']['rows']:,}", "", "sm"),
              kpi("Fraud", f"{P['dataset']['fraud']:,}", f"{P['dataset']['fraud_rate']:.2%} of transactions", "sm"),
              kpi("Accounts", f"{P['dataset']['accounts']:,}", "points in the graph", "sm"),
              kpi(f"{t('Louvain')} communities", f"{P['louvain']['communities']:,}", f"typical size {P['louvain']['median_size']} accounts", "sm")], 4)
        head("Why Louvain finds no fraud rings here",
             f"PaySim uses a new, single-use account for almost every fraud: only {P['louvain']['accounts_in_2plus_fraud']} of 3.28 million "
             "accounts appear in more than one fraud. There is no reused network to find, but fraud still has a shape: the money "
             "goes to accounts that are barely connected.")
        cen = {r["feature"]: r for r in P["centrality"]}
        mean = lambda s: float(s.split(" ")[0])
        c1, c2 = st.columns(2)
        leg = legend([(ORANGE, "receivers of fraud", "sq"), (BLUE, "receivers of normal payments", "sq")])
        with c1:
            bars([{"label": "Receiver's PageRank", "s": [(mean(cen["dest_pageRank"]["fraud"]), ORANGE, cen["dest_pageRank"]["fraud"]),
                                                         (mean(cen["dest_pageRank"]["legit"]), BLUE, cen["dest_pageRank"]["legit"])]}],
                 f"Receiver's {t('PageRank')}", "Average, with the median in brackets", leg=leg)
        with c2:
            bars([{"label": "Receiver's transactions", "s": [(mean(cen["dest_degree"]["fraud"]), ORANGE, cen["dest_degree"]["fraud"]),
                                                             (mean(cen["dest_degree"]["legit"]), BLUE, cen["dest_degree"]["legit"])]}],
                 f"Receiver's number of transactions ({t('Degree')})", "Average, with the median in brackets", leg=leg)
        insight([f"Graph features raise PR-AUC on future data from {ng['pr_auc']:.3f} to {g['pr_auc']:.3f}, far above the "
                 f"{P['rolling']['chance']:.3f} of random guessing.",
                 f"For a bank's investigators that means {fa_cut:.0%} fewer false alarms while catching {caught_up:.0%} more fraud.",
                 "The gain comes from PageRank and Degree, not from rings: fraud receivers have about half the connections of normal ones."])

    with tabs[1]:
        head("What a fraud receiver looks like in the network",
             "Each picture is a before snapshot: the account that received one test transaction, with the accounts it dealt with in "
             "steps 1-594. Dot colours show fraud in that earlier history only. The transaction being scored came later and is drawn "
             "as a thick dashed line. A fraud receiver is often blue because it had no fraud before: PaySim uses fresh, single-use accounts.")
        nb = load_json("paysim_neighbourhoods.json")
        if not nb:
            missing("The PaySim neighbourhood view", "python dashboard_prepare.py")
        else:
            short_lab = {"Fraud caught by the model": "Fraud caught", "Fraud missed by the model": "Fraud missed",
                         "Normal receiver (busy account)": "Busy normal account", "Normal receiver (typical)": "Typical normal account"}
            labs = [short_lab.get(n["label"], n["label"]) for n in nb]
            pk = pick("Example", labs, "nb_pick", labs[0])
            ex = nb[labs.index(pk)]
            tx = ex["transaction"]
            grid([kpi("Receiver's transactions", f"{tx['dest_degree']:,.0f}", "in steps 1-594", "sm"),
                  kpi("Receiver's PageRank", f"{tx['dest_pageRank']:.3f}", "normal receivers average 1.63", "sm"),
                  kpi("Fraud score", f"{tx['score_G']:.3f}", "from the graph model", "sm"),
                  kpi("Actually fraud?", "Yes" if tx["isFraud"] else "No", f"step {tx['step']}, amount {tx['amount']:,.0f}", "sm")], 4)
            network(ex["nodes"], ex["edges"], centre=ex["centre"],
                    scored={"sender": ex["sender"], "fraud": tx["isFraud"], "step": tx["step"], "amount": tx["amount"]})
            deg, prk = tx["dest_degree"], tx["dest_pageRank"]
            msg = {"Fraud caught": f"Before this payment the receiver had only {deg:,.0f} transaction(s) and a PageRank of {prk:.3f}: "
                                   "the barely used, single-purpose account typical of PaySim fraud. The model learned this pattern and flagged it.",
                   "Fraud missed": f"This fraud went to an account that looked normal and well used ({deg:,.0f} transactions, PageRank "
                                   f"{prk:.3f}). Its graph features pointed towards normal, so the model missed it. Fraud sent through "
                                   "accounts with a normal history is the method's main blind spot.",
                   "Busy normal account": f"A busy normal account ({deg:,.0f} transactions, PageRank {prk:.3f}). Many connections and a "
                                          "high PageRank push the score towards normal.",
                   "Typical normal account": f"A typical normal receiver ({deg:,.0f} transactions, PageRank {prk:.3f}): well connected, "
                                             "so the model scores it as normal."}
            insight(msg.get(pk, ""))
            note("Large neighbourhoods are sampled for display (up to 40 direct and 120 second-step transactions). "
                 "A sender with 0 transactions is a new account with no history.")

    with tabs[2]:
        with st.container(key="sections"):
            sec = pick("Section", ["Future data", "Algorithms", "Threshold", "Random split", "How the method was fixed"], "sec_paysim", "Future data")
        if sec == "Future data":
            head("Predicting future transactions",
                 f"The {t('rolling window')} test copies how a bank would work: learn from the past, predict the future, and only use "
                 "graph information that existed at the time.")
            H("<div class='rw'>"
              f"<div style='flex:475;background:{T['ink2']}'><b>Steps 1-475</b>Builds the graph only. 2.63 million transactions</div>"
              f"<div style='flex:150;background:{BLUE}'><b>476-594</b>Training. Features from the 1-475 graph</div>"
              f"<div style='flex:170;background:{ORANGE}'><b>595-743</b>Test, never seen. Features from the 1-594 graph</div></div>")
            m = pd.DataFrame(P["rolling"]["models"])
            if rw is not None and np.isnan(m.loc[1, "roc_auc"]):
                m.loc[1, "roc_auc"] = roc_auc_score(rw["isFraud"], rw["score_NG"])

            def model_card(r, name, sub, feature):
                rows = [("Precision", f"{r['precision']:.3f}"), ("Recall", f"{r['recall']:.3f}"), ("F1", f"{r['f1']:.3f}"),
                        ("ROC-AUC", "—" if np.isnan(r["roc_auc"]) else f"{r['roc_auc']:.3f}"),
                        ("False alarms", f"{r['false_alarms']:,}"), ("Fraud caught", f"{r['caught']:,}")]
                line = "rgba(255,255,255,.2)" if feature else "var(--line)"
                return (f"<div class='kpi {'feature' if feature else ''}' style='padding:1.2rem 1.4rem'>"
                        + ("<span class='tag'>Final model</span>" if feature else "")
                        + f"<div class='kl'>{name}</div><div class='kv'>{r['pr_auc']:.3f}<span>PR-AUC</span></div>"
                        + f"<div class='kn' style='margin-bottom:.5rem'>{sub}</div>"
                        + "".join(f"<div style='display:flex;justify-content:space-between;padding:.38rem 0;border-top:1px solid {line};"
                                  f"font-size:.9rem'><span>{a}</span><b>{b}</b></div>" for a, b in rows) + "</div>")
            H("<div class='grid' style='--c:2;margin-top:.9rem'>"
              + model_card(m.iloc[0], "With graph features (RW-G)", "PageRank, Degree and community size, plus amount, hour and type", True)
              + model_card(m.iloc[1], "Without graph features (RW-NG)", "Amount, hour and type only", False) + "</div>")
            if rw is not None:
                note("Threshold 0.5. The ROC-AUC of RW-NG was not recorded in the original run, so it is computed here from the re-created model.")
            c1, c2 = st.columns(2)
            with c1:
                imp = P["rolling"]["importance"]
                bars([{"label": FEATURE_TEXT[k], "s": [(v, BLUE, f"{v:.1%}")]} for k, v in imp.items()],
                     "What the graph model relies on", "Share of the model's total gain (XGBoost feature importance)")
            with c2:
                kr = P["rolling"]["by_receiver"]
                bars([{"label": ("Known receiver<small>already in the earlier graph</small>" if r["receiver"] == "Known"
                                 else "New receiver<small>first seen in the test period</small>"),
                       "s": [(r["RW-G"], BLUE, f"{r['RW-G']:.3f}"), (r["RW-NG"], ORANGE, f"{r['RW-NG']:.3f}")]} for r in kr],
                     "Known vs new receiving accounts", "PR-AUC for each kind of receiver", vmax=0.7,
                     leg=legend([(BLUE, "with graph", "sq"), (ORANGE, "without graph", "sq")]))
            note("<b>Why these features.</b> Type ranks first because fraud is far more common among TRANSFERs than CASH_OUTs. "
                 "Receiver's PageRank and transactions come next because fraud money goes to barely connected accounts. Hour helps because "
                 "normal activity follows a daily rhythm while fraud does not. Importance shows how much a feature is used, not whether it "
                 "raises or lowers the score; the Inspector tab shows that for single transactions.")
            insight([f"Graph features raise PR-AUC from {ng['pr_auc']:.3f} to {g['pr_auc']:.3f} and cut false alarms by {fa_cut:.0%}.",
                     "The graph helps for known and new receivers alike. New receivers score higher, because an account with no history "
                     "at all is itself a warning sign in PaySim.",
                     "Two of the top three features come from the graph."])
        elif sec == "Algorithms":
            head("Does the result depend on the algorithm or on luck?",
                 f"XGBoost, LightGBM and Random Forest, each with and without graph features, each run 5 times with different random "
                 f"seeds. Values are the average PR-AUC ± {t('std')} across the 5 runs. The test is repeated with the history graph "
                 "ending at step 475 (main test) and at step 400 (more training data).")
            rb = pd.DataFrame(P["robustness"])
            leg = legend([(BLUE, "with graph features", "sq"), (ORANGE, "without graph features", "sq")])
            c1, c2 = st.columns(2)
            for col, bd in zip((c1, c2), (475, 400)):
                d = rb[rb.boundary == bd]
                with col:
                    bars([{"label": r.algorithm, "s": [(r.graph, BLUE, f"{r.graph:.3f} ± {r.graph_std:.3f}"),
                                                       (r.no_graph, ORANGE, f"{r.no_graph:.3f} ± {r.no_graph_std:.3f}")]} for r in d.itertuples()],
                         f"History graph ends at step {bd}", "main test" if bd == 475 else "check with more training data", vmax=0.65, leg=leg)
            insight(["Graph features beat no graph features in all 30 runs, for every algorithm and both boundaries.",
                     "The gain (about +0.17 to +0.20) is far larger than the run-to-run spread (under 0.01), so it is not luck.",
                     "LightGBM is slightly higher than XGBoost (by 0.02-0.03). XGBoost stays the main model because it is the model set in "
                     "the objectives, and all three lead to the same conclusion: the benefit comes from the graph features."])
        elif sec == "Threshold":
            head("Choosing when to flag a transaction",
                 f"Every transaction gets a fraud score from 0 to 1. The {t('threshold')} decides which scores are flagged for review.")
            if rw is None:
                missing("The live threshold view", "python dashboard_prepare.py")
            else:
                threshold_panel(rw["isFraud"].values, {"With graph (RW-G)": rw["score_G"].values, "Without graph (RW-NG)": rw["score_NG"].values},
                                "thr_paysim")
            head("Thresholds chosen in advance", "A bank cannot see future labels, so the threshold was fixed on a validation slice "
                 "(steps 560-594) and then applied once to the test period.")
            th = pd.DataFrame(P["thresholds"])
            table(th, {"features": "Model", "rule": "How the threshold was chosen", "threshold": "Threshold", "precision": "Precision",
                       "recall": "Recall", "false_alarms": "False alarms"},
                  num=("Threshold", "Precision", "Recall", "False alarms"),
                  fmt={"Threshold": f3, "Precision": f3, "Recall": f3, "False alarms": ic})
            insight(["A recall target carries over: aiming for 75% on the validation slice gave 74.7% on future data.",
                     "A best-F1 threshold does not (it caught only 29% of fraud), because the share of fraud changes between periods: "
                     "1.5% in training, 3.2% in the test period.",
                     "At about 75% recall the graph model raises about 4 times fewer false alarms (4,636 against 20,171)."])
        elif sec == "Random split":
            head("The FYP1 evaluation and comparison experiments",
                 "Iterations 1-3 shuffled all transactions and split them 80:20 at random. That is optimistic: the graph was built "
                 "from every transaction, including the test ones. It was replaced by the rolling-window test, but it is kept here for comparison.")
            table(pd.DataFrame(P["iterations"]), {"model": "Iteration", "change": "What changed", "precision": "Precision", "recall": "Recall",
                                                  "f1": "F1", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"},
                  num=("Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"), fmt={k: f4 for k in ("Precision", "Recall", "F1", "PR-AUC", "ROC-AUC")})
            what = {"R1": "Baseline: the full Iteration 3 model", "R2": "Does the graph help? The same model without graph features",
                    "R3": "Why published scores reach 99%: keeps the balance columns that give the answer away",
                    "R4": "Does the algorithm matter? Random Forest", "R5": "Does the algorithm matter? LightGBM"}
            c1, c2 = st.columns([1.25, 1])
            with c1:
                bars([{"label": f"{r['id']}<small>{what[r['id']]}</small>",
                       "s": [(r["pr_auc"], ORANGE if r["id"] == "R3" else BLUE, f"{r['pr_auc']:.3f}")]} for r in P["comparisons"]],
                     "Comparison experiments R1-R5", "PR-AUC on the random split", vmax=1)
            with c2:
                bars([{"label": r["id"], "s": [(r["false_alarms"], ORANGE if r["id"] == "R2" else T["ink2"], f"{r['false_alarms']:,}")]}
                      for r in P["comparisons"]], "False alarms", "At threshold 0.5")
            rt = load_csv("paysim_random_test.csv.gz")
            if rt is not None:
                with st.expander("Try the threshold on the random split (Iteration 3)"):
                    threshold_panel(rt["isFraud"].values, {"Iteration 3": rt["score"].values}, "thr_random")
            insight(["Without graph features (R2) the model raises over 5 times more false alarms (16,666 against 3,074).",
                     "Keeping the balance columns (R3) gives 0.97 because those columns give the answer away. That is why they were removed, "
                     "and why many published 99% PaySim scores are not comparable.",
                     "XGBoost edges out Random Forest and LightGBM here, but the differences are small."])
        else:
            head("How the method was fixed",
                 "The first tests on future data scored worse than the random split. Each problem was found, explained and fixed in turn.")
            steps = [("Graph features from the whole timeline", f"A {t('leak')}: training features already contained information from the future test period."),
                     ("Leak removed, raw step number (T1)", "Every test step was higher than any step seen in training, so the model met values it had never seen."),
                     ("Hour of day instead of step (T3)", f"{t('Train-serve skew', 'train-serve skew')} remained: training features came from a graph "
                      "that already contained the training transactions themselves, test features did not."),
                     ("Rolling window (final, RW-G)", "Training and test features are both looked up from a graph of earlier steps only.")]
            H("<div class='tl'>" + "".join(
                f"<div class='it'><div class='step'>{i + 1}</div><div><b>{a}</b><p>{b}</p></div>"
                f"<div class='sc'>{r['pr_auc']:.3f}<small>PR-AUC</small></div></div>"
                for i, ((a, b), r) in enumerate(zip(steps, P["future_journey"]))) + "</div>")
            kn = P["known_new_t3t4"]
            after = {r["receiver"]: r["RW-G"] for r in P["rolling"]["by_receiver"]}
            c1, c2 = st.columns([1, 1])
            with c1:
                bars([{"label": f"{r['receiver']} receiver<small>{r['transactions']:,} transactions, {r['fraud']:,} fraud</small>",
                       "s": [(r["T4 no graph"], ORANGE, f"{r['T4 no graph']:.4f}"), (r["T3 graph"], T["ink2"], f"{r['T3 graph']:.4f}"),
                             (after[r["receiver"]], BLUE, f"{after[r['receiver']]:.4f}")]} for r in kn],
                     "Before and after the fix", "PR-AUC by receiving account on the test period", vmax=0.7,
                     leg=legend([(ORANGE, "without graph (T4)", "sq"), (T["ink2"], "with graph, before the fix (T3)", "sq"),
                                 (BLUE, "with graph, after the fix (RW-G)", "sq")]))
            with c2:
                insight(["Before the fix, the graph hurt known receivers (0.3025 with graph against 0.5075 without). If missing "
                         "history were the problem, it would have hurt new receivers instead.",
                         "That pointed to how training features were built. After the rolling window fix, known receivers rose from "
                         "0.3025 to 0.4604, and they make up about 77% of test transactions.",
                         "Overall PR-AUC therefore rose from 0.486 to 0.556. The new-receiver bars (0.6438 before, 0.5853 after) cover "
                         "only new accounts, so they are not comparable with the overall 0.556."])

    with tabs[3]:
        if rw is None:
            missing("The Prediction Inspector", "python dashboard_prepare.py")
        else:
            feats = (man or {}).get("parts", {}).get("paysim_rw", {}).get("features_G") or \
                ["origin_pageRank", "origin_degree", "origin_communitySize", "dest_pageRank", "dest_degree",
                 "dest_communitySize", "amount", "hour", "type_is_transfer"]
            inspector(rw, "isFraud", "score_G", load_model("model_paysim_rw_G.json"), feats,
                      "Model: the graph model (RW-G) on the unseen future data, steps 595-743.")

# ================================================================ SIMULATION
elif page == "Simulation (best case)":
    sim = load_csv("simulation_test.csv.gz")
    d = S["dataset"]
    f = S["models"][2]
    with tabs[0]:
        hero("Simulation: reused mule rings",
             "Built because PaySim never reuses mule accounts, so Louvain had nothing to find. Forty fraud rings each reuse 5-12 "
             "mule accounts across many laundering events, and each ring has its own victims. Rings are cleanly separated by design: "
             "the best case, to test whether the method works when rings exist.")
        H("<div style='height:.9rem'></div>")
        grid([kpi(f"{t('PR-AUC')} on an unseen batch", f"{f['pr_auc']:.3f}", "Final model (PageRank + Degree), new data from a different random seed", "feature"),
              kpi(f"{t('Fraud density', 'fraud density')} of fraud rings", "100%", "in every top fraud community Louvain found"),
              kpi("Fraud density of normal communities", "0%", "in the largest normal communities"),
              kpi("F1 on the unseen batch", f"{f['f1']:.3f}", f"{S['models'][1]['f1']:.3f} on its own test set: a small drop, so no overfitting")], 4)
        head("The data")
        grid([kpi("Normal accounts", f"{d['normal_accounts']:,}", "", "sm"),
              kpi("Fraud rings", f"{d['rings']}", f"{d['mules_per_ring']} reused {t('mule')}s each", "sm"),
              kpi("Training batch", f"{d['train_rows']:,}", f"{d['train_fraud']:,} fraud transactions", "sm"),
              kpi("Separate prediction batch", f"{d['pred_rows']:,}", f"{d['pred_fraud']:,} fraud transactions", "sm")], 4)
        insight(["When fraud rings reuse their mules, Louvain separates them perfectly: top fraud communities are 100% fraud, "
                 "large normal communities 0%.",
                 f"The final model scores {f['pr_auc']:.3f} on a batch it never saw, so what it learned carries over to new data.",
                 "This is the best case. The noisy simulation tests what happens when fraud is harder to separate."])
    with tabs[1]:
        head("Louvain's communities",
             "Each dot is an account and each line a transaction between two members of the same community. An all-orange cluster is a "
             "fraud ring; an all-blue group is ordinary customers paying each other.")
        gr = load_json("graph_simulation.json")
        if not gr:
            missing("The simulation graph view", "python dashboard_export_graph.py simulation")
        else:
            comms = gr["communities"]
            lab = [f"{'Fraud ring' if c['kind'] == 'fraud' else 'Normal community'} {c['community']}: "
                   f"{c['accounts']:,} accounts, {c['density']:.0%} fraud" for c in comms]
            i = st.selectbox("Community", range(len(comms)), format_func=lambda k: lab[k])
            cm = comms[i]
            grid([kpi("Accounts", f"{cm['accounts']:,}", f"{len(cm['nodes'])} shown" if cm["sampled"] else "", "sm"),
                  kpi("Transactions inside", f"{cm['tx']:,}", "between two members", "sm"),
                  kpi(t("Fraud density", "fraud density"), f"{cm['density']:.1%}", "", "sm")], 3)
            network(cm["nodes"], cm["edges"])
            insight("Fraud rings are small, dense and entirely orange, clearly apart from the large blue normal communities. "
                    "When mules are reused, the network itself exposes the ring." if cm["kind"] == "fraud" else
                    "Normal communities are large and contain no fraud. Fraud rings form their own separate groups, so Louvain isolates them.")
            note("Training batch, the same data as the report's Louvain table. Community numbers differ from the report because Louvain "
                 "labels communities differently on every run. Large communities are shown as a connected sample of 250 accounts.")
    with tabs[2]:
        head("Why the final model is not the highest-scoring one",
             "The model with community size scores almost perfectly, but only through a shortcut: in this dataset every ring is a small, "
             "separate community, so community size alone gives the answer. Real fraud is not that tidy.")
        m = S["models"]

        def sim_card(r, title, tag, tone):
            rows = [("Precision", f"{r['precision']:.3f}"), ("Recall", f"{r['recall']:.3f}"), ("F1", f"{r['f1']:.3f}"), ("ROC-AUC", f"{r['roc_auc']:.4f}")]
            line = "rgba(255,255,255,.2)" if tone == "feature" else "var(--line)"
            return (f"<div class='kpi {tone}' style='padding:1.2rem 1.4rem'>{tag}<div class='kl'>{title}</div>"
                    f"<div class='kv'>{r['pr_auc']:.3f}<span>PR-AUC</span></div>"
                    + "".join(f"<div style='display:flex;justify-content:space-between;padding:.38rem 0;border-top:1px solid {line};font-size:.9rem'>"
                              f"<span>{a}</span><b>{b}</b></div>" for a, b in rows) + "</div>")
        H("<div class='grid' style='--c:3'>"
          + sim_card(m[0], "With community size", "<span class='tag warn'>Shortcut</span>", "")
          + sim_card(m[1], "Final model: PageRank + Degree", "<span class='tag'>Final model</span>", "feature")
          + sim_card(m[2], "Final model on a separate batch", "<span class='tag warn' style='color:var(--blue);background:rgba(42,120,214,.12)'>New data</span>", "")
          + "</div>")
        c1, c2 = st.columns([1, 1.25])
        with c1:
            bars([{"label": FEATURE_TEXT[k], "s": [(v, ORANGE, f"{v:.1%}")]} for k, v in S["importance_full"].items()],
                 "With community size: what it relies on", "Almost entirely community size (99.95%)", vmax=1)
        with c2:
            bars([{"label": FEATURE_TEXT[k], "s": [(v, BLUE, f"{v:.1%}")]} for k, v in S["importance_final"].items()],
                 "Final model: what it relies on", "PageRank and Degree carry the signal", vmax=0.5)
        insight(["A near-perfect score from community size reflects how the dataset was built, not a method that will hold up.",
                 "The noisy simulation confirms it: when Louvain is re-run on messier data, community sizes change meaning and the "
                 "community-size model collapses (PR-AUC 0.08 at high noise).",
                 "The final model keeps 0.977 on unseen data without that shortcut, and relies on behaviour that makes sense: mules take "
                 "part in many transactions."])
        if sim is not None:
            head("Threshold on the separate batch")
            threshold_panel(sim["isFraud"].values, {"Final model": sim["score"].values}, "thr_sim",
                            foot="On this best-case data the curve stays near the top until high recall.")
    with tabs[3]:
        if sim is None:
            missing("The Prediction Inspector", "python dashboard_prepare.py")
        else:
            feats = (man or {}).get("parts", {}).get("simulation", {}).get("features") or \
                ["origin_pageRank", "origin_degree", "dest_pageRank", "dest_degree", "amount", "step", "type_is_transfer"]
            inspector(sim, "isFraud", "score", load_model("model_simulation.json"), feats,
                      "Model: the final simulation model (PageRank + Degree) on the separate prediction batch.")

# ================================================================ NOISY
else:
    L = N["levels"][level]
    nz = load_csv(f"noisy_{level}_test.csv.gz")
    MEANING = {
        "Victims who are normal customers": "Victims also make ordinary payments, which links the ring to the normal network.",
        "Everyday transactions per mule": "Mules also pay bills and shop, so they look like normal customers.",
        "Laundering shapes": "Which money-moving patterns the rings use (pictured below).",
        "Fraud events with missing labels": "Some real fraud is never reported, so it is labelled normal in training. Results use the true labels.",
        "Mules replaced per week": "Old mule accounts are dropped and new ones recruited, so a ring's accounts change over time.",
    }
    with tabs[0]:
        hero(f"Noisy simulation, {level} level",
             "The same rings, made harder to see. Only fraud behaviour changes between the four levels, following published research on "
             f"how fraud hides. Tested with the same {t('rolling window')} as PaySim. Amount, time and type carry no fraud signal by "
             "design, so only the network can separate fraud.")
        H("<div style='height:.9rem'></div>")
        grid([kpi(f"{t('PR-AUC')}, final model", f"{L['G-noCS']:.3f}", f"About {L['G-noCS'] / L['chance']:.0f} times the chance level of {L['chance']:.3f}", "feature"),
              kpi(f"{t('Fraud isolated', 'fraud isolated')} by Louvain",
                  f"{L['louvain_share']:.1%}", f"Average of 5 {t('reload')}s; from {L['louvain_min']:.0%} to {L['louvain_max']:.0%}"),
              kpi("Transactions", f"{L['tx']:,}", "steps 1-743"),
              kpi("True fraud", f"{L['fraud']:,}", "including fraud with missing labels")], 4)
        head("What changes at this level", "Five sources of noise, each based on a published description of how real fraud behaves.")
        sd = pd.DataFrame(N["settings"])
        sd["meaning"] = sd["noise source"].map(MEANING)
        table(sd, {"noise source": "Noise source", level: f"At {level}", "meaning": "What it means", "basis": "Source"}, num=(f"At {level}",))
        with st.expander("Compare all four levels"):
            table(sd, {"noise source": "Noise source", "clean": "Clean", "low": "Low", "medium": "Medium", "high": "High", "basis": "Source"})
        head("The five laundering shapes", "Orange dots are mule accounts and arrows show money moving. Typologies from Altman et al. (2023).")
        H(shapes_html())
        note("<b>Mules replaced per week.</b> Firms routinely find and close mule accounts: UK firms closed 238,396 suspected mule "
             "accounts in 2025, nearly half within a year of opening (FCA, 2026), so rings cannot keep the same mules for ever. "
             "No published source gives a weekly rate, so the rates used (5%, 15%, 30%) are a design choice to stress-test the method.")
        insight([f"At the {level} level, {L['louvain_share']:.0%} of fraud sits in communities that are mostly fraud (between "
                 f"{L['louvain_min']:.0%} and {L['louvain_max']:.0%} across 5 reloads of the same data).",
                 f"The final model scores {L['G-noCS']:.3f}, about {L['G-noCS'] / L['chance']:.0f} times what random guessing would score."])
    with tabs[1]:
        head(f"Louvain's communities at the {level} level",
             "An all-orange cluster is a ring Louvain kept separate. Orange mixed into blue means fraud accounts were absorbed into a large "
             f"normal community, hidden among normal activity. Switch the noise level at the top to watch rings merge as {t('camouflage')} grows.")
        gr = load_json(f"graph_{level}.json")
        if not gr:
            missing(f"The {level} graph view", f"python noisy_pipeline.py {level}  then  python dashboard_export_graph.py {level}")
        else:
            comms = gr["communities"]
            lab = [f"{'Top fraud' if c['kind'] == 'fraud' else 'Large normal'} community {c['community']}: "
                   f"{c['accounts']:,} accounts, {c['density']:.1%} fraud" for c in comms]
            i = st.selectbox("Community", range(len(comms)), format_func=lambda k: lab[k], key=f"comm_{level}")
            cm = comms[i]
            grid([kpi("Accounts", f"{cm['accounts']:,}", f"{len(cm['nodes'])} shown" if cm["sampled"] else "", "sm"),
                  kpi("Transactions inside", f"{cm['tx']:,}", "between two members", "sm"),
                  kpi(t("Fraud density", "fraud density"), f"{cm['density']:.1%}", "", "sm")], 3)
            network(cm["nodes"], cm["edges"])
            insight("At the clean level, fraud rings form their own all-fraud communities, as in the best-case simulation." if level == "clean"
                    else f"At the {level} level, fraud accounts are spread through large communities where most activity is normal (this one "
                    f"is {cm['density']:.1%} fraud). Louvain can no longer isolate the rings: this is what camouflage does.")
            note(f"One Louvain run. Louvain's result changes between reloads of the same data: at this level {L['louvain_min']:.0%} to "
                 f"{L['louvain_max']:.0%} of fraud was isolated across 5 reloads. Large communities show all fraud accounts (up to 150), then a sample of normal ones.")
    with tabs[2]:
        with st.container(key="sections"):
            sec = pick("Section", ["Across noise levels", "One noise source at a time", "Why community size was dropped", "Threshold"],
                       "sec_noisy", "Across noise levels")
        if sec == "Across noise levels":
            head("How the method holds up as fraud hides",
                 "Three models on future data at each level: the final model (PageRank + Degree), the model that also uses community size, "
                 f"and a model without graph features. PR-AUC is the average of 5 random seeds; Louvain bars are the average of 5 {t('reload')}s, "
                 "with the black marker spanning the lowest to the highest run.")
            df = pd.DataFrame([{"level": k, **v} for k, v in N["levels"].items()])
            long = df.melt(id_vars="level", value_vars=["G-noCS", "G", "NG", "chance"], var_name="model", value_name="pr_auc")
            names = {"G-noCS": "PageRank + Degree (final)", "G": "With community size", "NG": "No graph features", "chance": "Random guessing"}
            long["model"] = long["model"].map(names)
            dom = list(names.values())
            c1, c2 = st.columns([1.15, 1], gap="medium")
            with c1:
                with card("lv_line"):
                    H("<div class='bars' style='padding:0'><div class='bt'>Detection on future data</div><div class='bs'>PR-AUC, average of 5 seeds</div>"
                      + legend([(BLUE, dom[0], "ln"), (AQUA, dom[1], "ln"), (ORANGE, dom[2], "ln"), (GREY, dom[3], "ln")]) + "</div>")
                    ch = alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=80, filled=True), strokeWidth=2.5).encode(
                        x=alt.X("level:N", sort=LEVELS, title=None, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("pr_auc:Q", title=None, scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("model:N", scale=alt.Scale(domain=dom, range=[BLUE, AQUA, ORANGE, GREY]), legend=None),
                        strokeDash=alt.condition(alt.datum.model == "Random guessing", alt.value([4, 4]), alt.value([0])),
                        tooltip=["level", "model", alt.Tooltip("pr_auc:Q", format=".3f")])
                    st.altair_chart(style(ch.properties(height=262)), use_container_width=True)
            with c2:
                bars([{"label": r.level.capitalize(), "s": [(r.louvain_share, BLUE, f"{r.louvain_share:.1%}")], "lo": r.louvain_min, "hi": r.louvain_max}
                      for r in df.itertuples()], f"{t('Fraud isolated', 'fraud isolated')} by Louvain",
                     "Bar = average of 5 reloads, marker = lowest to highest", vmax=1)
            table(df, {"level": "Level", "G-noCS": "Final model", "G": "With community size", "NG": "No graph", "chance": "Random guessing",
                       "louvain_share": "Louvain isolates"},
                  num=("Final model", "With community size", "No graph", "Random guessing", "Louvain isolates"),
                  fmt={"Final model": f3, "With community size": f3, "No graph": f3, "Random guessing": f3, "Louvain isolates": pc,
                       "Level": lambda v: v.capitalize()})
            hi = N["levels"]["high"]
            insight(["Without graph features the model stays at the level of random guessing at every level: by design, only the network carries the signal.",
                     "Louvain's isolation falls from 87% at clean to 7% at high noise, and its result varies widely between reloads of the same data.",
                     f"The final model falls from 0.910 to about 0.3-0.4 but stays far above random guessing ({hi['G-noCS']:.3f} against {hi['chance']:.3f} at high).",
                     f"The model with community size collapses at high noise ({hi['G']:.3f}), which is why the final model leaves it out."])
        elif sec == "One noise source at a time":
            head("Which noise source does the damage?",
                 "Each row switches on one noise source at its high setting with everything else clean. Clean is the starting point. "
                 "The last two rows use full high noise with a single laundering shape, to see which shapes Louvain can still find.")
            of = pd.DataFrame(N["ofat"])
            c1, c2 = st.columns(2)
            with c1:
                bars([{"label": r.run, "s": [(r.louvain_share, BLUE, f"{r.louvain_share:.0%}")], "lo": r.louvain_min, "hi": r.louvain_max}
                      for r in of.itertuples()], f"{t('Fraud isolated', 'fraud isolated')} by Louvain",
                     "Average of 5 reloads; marker = lowest to highest. Clean: 87%, from 85% to 90%", vmax=1)
            with c2:
                bars([{"label": r["run"], "s": [(r["G-noCS"], BLUE, f"{r['G-noCS']:.3f}")]} for r in of.to_dict("records")],
                     "Final model (PageRank + Degree)", "PR-AUC, average of 5 seeds. Random guessing scores about 0.01", vmax=1)
            insight(["Only camouflage breaks Louvain: shared victims (16%) and mule everyday transactions (8%), against 87% at clean. "
                     "The other noise sources leave it at 85-89%.",
                     "Camouflage also makes Louvain unstable: with mule everyday transactions, reloads of the same data range from 0% to 21%.",
                     "Under heavy noise, fan-in rings never survive, while tightly connected scatter-gather rings survive in some runs (5-51%).",
                     "Among single sources the final model is hurt most when mules behave like normal customers (0.698). With full high "
                     "noise and only fan-in rings it falls to 0.178, still well above random guessing."])
        elif sec == "Why community size was dropped":
            head("Community size changes meaning when Louvain is re-run",
                 "Louvain runs twice: on steps 1-475 for training features and on steps 1-594 for test features. A rule like "
                 "\"fraud sits in communities of about this size\" only works if sizes mean the same thing in both runs.")
            cs = pd.DataFrame(N["community_size_check"])
            table(cs, {"level": "Level", "changed_2x": "Accounts whose community size changed more than 2x",
                       "normal_median": "Typical normal community size, training to test",
                       "normal_in_fraud_range": "Normal transactions in the fraud size range, training to test",
                       "G_false_alarms": "False alarms of the community-size model"},
                  num=("Accounts whose community size changed more than 2x", "False alarms of the community-size model"),
                  fmt={"Accounts whose community size changed more than 2x": p0, "False alarms of the community-size model": ic,
                       "Typical normal community size, training to test": lambda v: v.replace("->", "to"),
                       "Normal transactions in the fraud size range, training to test": lambda v: v.replace("->", "to")})
            note("<b>Fraud size range:</b> the range of community sizes where fraud sat during training. A normal transaction inside it "
                 "would be flagged by a size rule.")
            insight(["At clean, sizes barely move between the two runs (6% of accounts) and the size rule raises no false alarms.",
                     "With noise, 43-72% of accounts change community size more than 2x, and typical normal communities grow 2-3 times.",
                     "At high noise every normal test transaction falls in the fraud size range, giving 73,860 false alarms.",
                     "Community size does not keep a stable meaning, so the final model uses PageRank and Degree only."])
        else:
            head(f"Threshold at the {level} level",
                 f"All three models, scored against the true labels, including fraud whose label was missing in training ({t('label noise')}).")
            if nz is None:
                missing("The live threshold view", "python dashboard_prepare.py")
            else:
                threshold_panel(nz["isFraud_true"].values, {"PageRank + Degree (final)": nz["score_G-noCS"].values,
                                                            "No graph features": nz["score_NG"].values,
                                                            "With community size": nz["score_G"].values}, f"thr_{level}",
                                foot="Without graph features the curve stays near the bottom at every threshold: fraud cannot be separated without the network.")
    with tabs[3]:
        if nz is None:
            missing("The Prediction Inspector", "python dashboard_prepare.py")
        else:
            feats = (man or {}).get("parts", {}).get(f"noisy_{level}", {}).get("features") or \
                ["origin_pageRank", "origin_degree", "dest_pageRank", "dest_degree", "amount", "hour", "type_is_transfer"]
            inspector(nz, "isFraud_true", "score_G-noCS", load_model(f"model_noisy_{level}.json"), feats,
                      f"Model: the final model (PageRank + Degree) at the {level} level, scored against the true labels.")
