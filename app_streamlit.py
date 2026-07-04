# =====================================================================
# IMACID — Interface Streamlit Professionnelle
# Prediction P2O5 — Deploiement Cloud
# =====================================================================
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import requests
import time
import io
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="IMACID — Prediction P2O5",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }
.stApp { background: radial-gradient(ellipse at top, #0A1628 0%, #020408 70%); }
.kpi-card {
    background: linear-gradient(135deg, #0D1B2A, #111D2E);
    border: 1px solid rgba(240,180,41,0.3);
    border-radius: 12px; padding: 20px;
    text-align: center; margin: 8px 0;
}
.kpi-value { font-family: Orbitron, monospace; font-size: 2rem; color: #F0B429; }
.kpi-label { color: #8B949E; font-size: 12px; letter-spacing: 2px; text-transform: uppercase; }
.stButton > button {
    background: linear-gradient(135deg, #F0B429, #D4940A) !important;
    color: #020408 !important; font-family: Orbitron, monospace !important;
    font-weight: 700 !important; letter-spacing: 2px !important;
    border-radius: 8px !important; border: none !important;
    box-shadow: 0 0 20px rgba(240,180,41,0.35) !important;
}
[data-testid="stMetricValue"] { font-family: Orbitron, monospace !important; color: #F0B429 !important; }
.sidebar .sidebar-content { background: #0D1B2A; }
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────
CHIM = ["Densite","TS","CO2","SiO2(T)","SiO2(R)","CaO","SO4","F-",
        "Fe2O3","Al2O3","K2O","MgO","Na2O","Cd"]
GRAN = [">500",">400",">315",">250",">160",">125",">80",">40"]
ALL  = CHIM + GRAN
RMSE = 0.1978

DEFAULTS = {
    "Densite":1.65,"TS":65.0,"CO2":5.2,"SiO2(T)":14.5,"SiO2(R)":13.0,
    "CaO":48.0,"SO4":2.8,"F-":3.1,"Fe2O3":2.9,"Al2O3":4.0,
    "K2O":0.78,"MgO":1.45,"Na2O":0.65,"Cd":0.001,
    ">500":2.0,">400":5.0,">315":12.0,">250":25.0,
    ">160":55.0,">125":70.0,">80":85.0,">40":98.0,
}
RANGES = {
    "Densite":(1.2,2.2),"TS":(40,85),"CO2":(0,15),"SiO2(T)":(5,25),
    "SiO2(R)":(4,22),"CaO":(35,58),"SO4":(0.5,6),"F-":(1,6),
    "Fe2O3":(1.5,5),"Al2O3":(1.5,7),"K2O":(0.4,1.2),"MgO":(0.8,3.5),
    "Na2O":(0.2,1.5),"Cd":(0,0.01),
    ">500":(0,20),">400":(0,30),">315":(0,40),">250":(0,60),
    ">160":(10,90),">125":(30,95),">80":(50,99),">40":(80,100),
}
QUAL_COLORS = {
    "PREMIUM":"#2ECC71","STANDARD":"#F0B429","LIMITE":"#E67E22","ALERTE":"#E74C3C"
}

# ── Chargement modele ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path("IMACID_Resultats/modeles/XGBoost_champion.joblib")
    if not model_path.exists():
        st.error("Modele introuvable : " + str(model_path.resolve()))
        st.stop()
    return joblib.load(model_path)

model = load_model()

def predict(vals_dict):
    X = np.array([[vals_dict.get(f, DEFAULTS[f]) for f in ALL]])
    p = float(model.predict(X)[0])
    if   p >= 31.0: q = "PREMIUM"
    elif p >= 29.5: q = "STANDARD"
    elif p >= 28.0: q = "LIMITE"
    else:           q = "ALERTE"
    return round(p, 4), q, round(p - 1.96*RMSE, 4), round(p + 1.96*RMSE, 4)

def pred_dataframe(df_in):
    df = df_in.copy()
    col_map = {c: f for c in df.columns for f in ALL if c.strip().lower() == f.lower()}
    df = df.rename(columns=col_map)
    for f in ALL:
        if f not in df.columns:
            df[f] = DEFAULTS[f]
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(DEFAULTS[f])
    X = df[ALL].values.astype(float)
    preds = np.array(model.predict(X)).ravel()
    df["P2O5_Predit"]  = preds.round(4)
    df["IC_Low_95"]    = (preds - 1.96*RMSE).round(4)
    df["IC_High_95"]   = (preds + 1.96*RMSE).round(4)
    df["Qualite"]      = pd.cut(preds,
        bins=[-np.inf, 28.0, 29.5, 31.0, np.inf],
        labels=["ALERTE","LIMITE","STANDARD","PREMIUM"])
    df["Timestamp"]    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:28px 0 12px;">
<h1 style="font-family:Orbitron,monospace; color:#F0B429; font-size:2.2rem;
letter-spacing:6px; text-shadow:0 0 30px rgba(240,180,41,0.5); margin:0;">
IMACID</h1>
<p style="color:#8B949E; font-family:Exo2; letter-spacing:3px; font-size:12px;
text-transform:uppercase; margin:6px 0 0;">
Systeme de Prediction Automatique P2O5 — XGBoost</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Modele", "XGBoost")
col2.metric("R2", "0.8694")
col3.metric("RMSE", "0.1978%")
col4.metric("RPD", "2.77")

st.markdown("---")

# ── Sidebar navigation ────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:12px 0;">
    <span style="font-family:Orbitron; color:#F0B429; font-size:14px; letter-spacing:2px;">
    NAVIGATION</span>
    </div>
    """, unsafe_allow_html=True)
    mode = st.radio("", [
        "Saisie manuelle",
        "Import CSV / Excel",
        "Historique session",
        "A propos"
    ], label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════════
# MODE 1 — SAISIE MANUELLE
# ══════════════════════════════════════════════════════════════════
if mode == "Saisie manuelle":
    st.subheader("Saisie des analyses")

    with st.form("form_predict"):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Analyses Chimiques**")
            vals = {}
            for feat in CHIM:
                lo, hi = RANGES[feat]
                vals[feat] = st.number_input(
                    feat, value=float(DEFAULTS[feat]),
                    min_value=float(lo), max_value=float(hi),
                    step=float(round((hi-lo)/200, 6)),
                    format="%.4f"
                )

        with c2:
            st.markdown("**Granulometrie (%)**")
            for feat in GRAN:
                lo, hi = RANGES[feat]
                vals[feat] = st.number_input(
                    feat, value=float(DEFAULTS[feat]),
                    min_value=float(lo), max_value=float(hi),
                    step=0.1, format="%.2f"
                )

        submitted = st.form_submit_button(
            "PREDIRE LA TENEUR EN P2O5",
            use_container_width=True
        )

    if submitted:
        p2o5, qual, ic_lo, ic_hi = predict(vals)
        qc = QUAL_COLORS[qual]

        st.markdown(
            "<div class='kpi-card'>"
            "<div class='kpi-value'>" + str(p2o5) + " %</div>"
            "<div class='kpi-label'>" + qual + " — IC 95% [" + str(ic_lo) + " — " + str(ic_hi) + "]</div>"
            "</div>",
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("P2O5 Predit", str(p2o5) + " %")
        c2.metric("Qualite", qual)
        c3.metric("IC 95%", "[" + str(ic_lo) + " ; " + str(ic_hi) + "]")

        # Jauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p2o5,
            number=dict(suffix="%", font=dict(color=qc, size=40)),
            gauge=dict(
                axis=dict(range=[27, 33], tickcolor="#8B949E"),
                bar=dict(color=qc, thickness=0.3),
                bgcolor="#0D1B2A",
                bordercolor="#1E3A5F",
                steps=[
                    dict(range=[27,   28.0], color="#1A0808"),
                    dict(range=[28.0, 29.5], color="#1A0D00"),
                    dict(range=[29.5, 31.0], color="#0D1B00"),
                    dict(range=[31.0, 33.0], color="#071A0E"),
                ],
                threshold=dict(line=dict(color=qc, width=4), value=p2o5)
            ),
            title=dict(text="P2O5 Predit", font=dict(color="#F0B429", size=16))
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#050A0F", font_color="#F0F6FC", height=320,
            margin=dict(t=60, b=20, l=30, r=30)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Sauvegarder dans l historique session
        if "historique" not in st.session_state:
            st.session_state["historique"] = []
        st.session_state["historique"].append({
            "Heure":      datetime.now().strftime("%H:%M:%S"),
            "P2O5 (%)":   p2o5,
            "Qualite":    qual,
            "IC_Low":     ic_lo,
            "IC_High":    ic_hi,
        })

# ══════════════════════════════════════════════════════════════════
# MODE 2 — IMPORT CSV / EXCEL
# ══════════════════════════════════════════════════════════════════
elif mode == "Import CSV / Excel":
    st.subheader("Import fichier analyses")

    uploaded = st.file_uploader(
        "Deposer votre fichier ici",
        type=["csv", "xlsx", "xls"],
        help="1 ligne = 1 echantillon | colonnes = variables analytiques"
    )

    if uploaded:
        with st.spinner("Traitement en cours..."):
            ext = uploaded.name.split(".")[-1].lower()
            if ext == "csv":
                df_in = pd.read_csv(uploaded, sep=None, engine="python",
                                    decimal=",", encoding="utf-8-sig")
            else:
                df_in = pd.read_excel(uploaded)

            df_res = pred_dataframe(df_in)

        n = len(df_res)
        n_prem = (df_res["Qualite"] == "PREMIUM").sum()
        n_std  = (df_res["Qualite"] == "STANDARD").sum()
        n_lim  = (df_res["Qualite"] == "LIMITE").sum()
        n_alt  = (df_res["Qualite"] == "ALERTE").sum()

        st.success(str(n) + " echantillons traites")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PREMIUM",  str(n_prem), delta=None)
        c2.metric("STANDARD", str(n_std))
        c3.metric("LIMITE",   str(n_lim))
        c4.metric("ALERTE",   str(n_alt))

        # Graphique barres
        qual_colors_list = [
            QUAL_COLORS.get(str(q), "#8B949E") for q in df_res["Qualite"]
        ]
        fig_bar = go.Figure(go.Bar(
            x=list(range(n)),
            y=df_res["P2O5_Predit"].tolist(),
            marker_color=qual_colors_list,
            name="P2O5"
        ))
        for lv, col, lbl in [
            (31.0, "#2ECC71", "PREMIUM"),
            (29.5, "#F0B429", "STANDARD"),
            (28.0, "#E74C3C", "ALERTE")
        ]:
            fig_bar.add_hline(y=lv, line_dash="dot",
                              line_color=col, annotation_text=lbl)
        fig_bar.update_layout(
            paper_bgcolor="#050A0F", plot_bgcolor="#0D1B2A",
            font_color="#F0F6FC", height=380,
            title=dict(text="P2O5 par echantillon",
                       font=dict(color="#F0B429")),
            xaxis=dict(gridcolor="#1E3A5F"),
            yaxis=dict(gridcolor="#1E3A5F", range=[27, 33]),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Tableau resultats
        cols_show = ["P2O5_Predit", "IC_Low_95", "IC_High_95", "Qualite", "Timestamp"]
        st.dataframe(df_res[cols_show], use_container_width=True)

        # Telechargement
        csv_out = df_res.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Telecharger les resultats CSV",
            data=csv_out,
            file_name="predictions_" + datetime.now().strftime("%Y%m%d_%H%M") + ".csv",
            mime="text/csv",
            use_container_width=True
        )

# ══════════════════════════════════════════════════════════════════
# MODE 3 — HISTORIQUE
# ══════════════════════════════════════════════════════════════════
elif mode == "Historique session":
    st.subheader("Historique des predictions")
    if "historique" not in st.session_state or not st.session_state["historique"]:
        st.info("Aucune prediction effectuee dans cette session.")
    else:
        df_hist = pd.DataFrame(st.session_state["historique"])
        st.dataframe(df_hist, use_container_width=True)

        fig_hist = go.Figure(go.Scatter(
            x=df_hist["Heure"].tolist(),
            y=df_hist["P2O5 (%)"].tolist(),
            mode="lines+markers",
            line=dict(color="#F0B429", width=2),
            marker=dict(size=8, color="#F0B429"),
            fill="tozeroy",
            fillcolor="rgba(240,180,41,0.06)"
        ))
        fig_hist.update_layout(
            paper_bgcolor="#050A0F", plot_bgcolor="#0D1B2A",
            font_color="#F0F6FC", height=320,
            title=dict(text="Evolution P2O5 session",
                       font=dict(color="#F0B429")),
            xaxis=dict(gridcolor="#1E3A5F"),
            yaxis=dict(gridcolor="#1E3A5F", range=[27, 33])
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        if st.button("Effacer l historique"):
            st.session_state["historique"] = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════
# MODE 4 — A PROPOS
# ══════════════════════════════════════════════════════════════════
elif mode == "A propos":
    st.subheader("A propos du modele")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Modele :** XGBoost Regressor
**Configuration :** Nettoyé
**Dataset :** 366 echantillons IMACID

| Metrique | Valeur |
|----------|--------|
| R2 Test  | 0.8694 |
| RMSE     | 0.1978% |
| MAE      | 0.1639% |
| RPD      | 2.77 |
| Biais    | 0.0153% |
        """)
    with c2:
        st.markdown("""
**Seuils qualite :**

| Classe | P2O5 |
|--------|------|
| PREMIUM | >= 31.0% |
| STANDARD | 29.5 — 31.0% |
| LIMITE | 28.0 — 29.5% |
| ALERTE | < 28.0% |

**IC 95% :** prediction +/- 0.3877%
        """)
