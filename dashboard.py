"""
dashboard.py  —  Lahore AQI Forecasting Dashboard
══════════════════════════════════════════════════
Features:
  • 3-day AQI forecast (72 hours)
  • Multiple models: XGBoost, Random Forest
  • SHAP feature importance explanations
  • Hazardous AQI alerts
  • Historical trends & EDA
  • Model performance registry
  • Real-time data from MongoDB

Run:
    pip install streamlit plotly shap pymongo[srv] xgboost scikit-learn joblib python-dotenv
    streamlit run dashboard.py
"""

from dotenv import load_dotenv
load_dotenv()

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
from datetime import datetime, timedelta
from pathlib import Path
import joblib

st.set_page_config(page_title="AQI Forecast — Lahore", page_icon="🌫️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#080c14;}
.block-container{padding:1.5rem 2rem;}
h1,h2,h3{font-family:'Space Mono',monospace!important;}
.aqi-hero{background:linear-gradient(135deg,#0f1923 0%,#1a2535 100%);border:1px solid rgba(255,255,255,0.08);border-radius:20px;padding:28px 24px;text-align:center;}
.aqi-number{font-family:'Space Mono',monospace;font-size:5rem;font-weight:700;line-height:1;margin:0;}
.aqi-label{font-size:1.1rem;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-top:8px;}
.aqi-city{color:rgba(255,255,255,0.4);font-size:0.85rem;margin-top:6px;letter-spacing:1px;}
.metric-card{background:#0f1923;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:18px;text-align:center;}
.metric-val{font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;}
.metric-name{color:rgba(255,255,255,0.45);font-size:0.78rem;letter-spacing:1px;text-transform:uppercase;margin-top:4px;}
.alert-haz{background:linear-gradient(90deg,#7e0023,#b71c1c);border-radius:12px;padding:14px 20px;color:white;font-weight:600;}
.alert-warn{background:linear-gradient(90deg,#e65100,#f57c00);border-radius:12px;padding:14px 20px;color:white;font-weight:600;}
.fcard{background:#0f1923;border:1px solid rgba(255,255,255,0.07);border-radius:16px;padding:20px 16px;text-align:center;}
.fday{font-size:0.8rem;color:rgba(255,255,255,0.45);letter-spacing:1px;text-transform:uppercase;}
.faqi{font-family:'Space Mono',monospace;font-size:2.4rem;font-weight:700;margin:8px 0 4px;}
.flabel{font-size:0.8rem;font-weight:600;letter-spacing:1px;}
div[data-testid="stSidebar"]{background:#0a0f1a;border-right:1px solid rgba(255,255,255,0.06);}
</style>
""", unsafe_allow_html=True)

AQI_LEVELS = [
    (50,  "#00e676", "Good",                    "Air quality is satisfactory."),
    (100, "#ffea00", "Moderate",                "Acceptable; some pollutants may be a concern."),
    (150, "#ff9100", "Unhealthy for Sensitive", "Sensitive groups may experience health effects."),
    (200, "#ff1744", "Unhealthy",               "Everyone may begin to experience health effects."),
    (300, "#d500f9", "Very Unhealthy",          "Health alert: everyone may experience serious effects."),
    (999, "#b71c1c", "Hazardous",               "Health warning of emergency conditions."),
]

def aqi_info(val):
    for threshold, color, label, desc in AQI_LEVELS:
        if val <= threshold:
            return color, label, desc
    return "#b71c1c", "Hazardous", "Emergency conditions."

@st.cache_resource
def get_db():
    uri = os.getenv("MONGODB_URI", "")
    if not uri:
        st.error("❌ Set MONGODB_URI in your .env file")
        st.stop()
    return MongoClient(uri)["aqi_pipeline"]

@st.cache_data(ttl=300)
def load_features():
    docs = list(get_db()["weather_aqi_features"].find({}, {"_id": 0}))
    df = pd.DataFrame(docs)
    # df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    # df = df.dropna(subset=["aqi"]).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[df["timestamp"] > pd.Timestamp("2020-01-01", tz="UTC")]  # remove bad dates
    df = df.dropna(subset=["aqi"]).sort_values("timestamp").reset_index(drop=True)
    return df

@st.cache_data(ttl=60)
def load_registry():
    docs = list(get_db()["model_registry"].find({}, {"_id": 0}).sort("trained_at", -1).limit(20))
    return pd.DataFrame(docs) if docs else pd.DataFrame()

@st.cache_resource
def load_models():
    out = Path("aqi_model_outputs")
    models = {}
    for name, fname in [("XGBoost","aqi_xgb_model.pkl"),("Random Forest","model_reg_target_aqi_current_RandomForest.pkl")]:
        p = out / fname
        if p.exists():
            try: models[name] = joblib.load(p)
            except: pass
    imputer = joblib.load(out/"imputer.pkl") if (out/"imputer.pkl").exists() else None
    feat_cols = json.load(open(out/"feature_cols.json")) if (out/"feature_cols.json").exists() else []
    return models, imputer, feat_cols

def prepare_X(df, feat_cols):
    from sklearn.preprocessing import LabelEncoder
    d = df.copy()
    if "city" in d.columns:
        le = LabelEncoder()
        d["city_encoded"] = le.fit_transform(d["city"].astype(str))
    drop = ["timestamp","city","dominant_poll","aqi_cat_label","aqi_cat_ordinal","aqi"]
    d = d.drop(columns=[c for c in drop if c in d.columns], errors="ignore")
    d = d.select_dtypes(include=[np.number])
    for col in feat_cols:
        if col not in d.columns: d[col] = 0
    return d[feat_cols]

def make_forecast(df, model, imputer, feat_cols, hours=72):
    base = df.tail(72).copy()
    base["hour"]        = base["timestamp"].dt.hour
    base["day_of_week"] = base["timestamp"].dt.dayofweek
    base["month"]       = base["timestamp"].dt.month
    base["day_of_year"] = base["timestamp"].dt.dayofyear
    base["is_weekend"]  = (base["day_of_week"] >= 5).astype(int)
    base["is_night"]    = ((base["hour"] >= 22) | (base["hour"] <= 6)).astype(int)
    X = prepare_X(base, feat_cols)
    X_imp = imputer.transform(X) if imputer else X.values
    preds = np.clip(model.predict(X_imp), 0, 500)
    last_ts = df["timestamp"].iloc[-1]
    future_ts = [last_ts + timedelta(hours=i+1) for i in range(hours)]
    pr = list(preds[-min(len(preds),hours):])
    while len(pr) < hours: pr.append(pr[-1] * np.random.uniform(0.97, 1.03))
    return pd.DataFrame({"timestamp": future_ts[:hours], "aqi": np.clip(pr[:hours], 0, 500)})

# Sidebar
with st.sidebar:
    st.markdown("### 🌫️ AQI FORECAST")
    st.markdown("<p style='color:rgba(255,255,255,0.4);font-size:0.8rem;margin-top:-10px'>Lahore, Pakistan</p>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("", ["🏠  Overview","🔮  3-Day Forecast","📈  Historical EDA","🤖  Model Registry","🔍  SHAP Explainer"], label_visibility="collapsed")
    st.divider()
    models_dict, imputer, feat_cols = load_models()
    model_choice = st.selectbox("Forecast Model", list(models_dict.keys()) if models_dict else ["XGBoost"])
    st.divider()
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

with st.spinner("Loading from MongoDB..."):
    df = load_features()

active_model = models_dict.get(model_choice) if models_dict else None
PLOT_CFG = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,35,0.8)", margin=dict(l=0,r=0,t=10,b=0))

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
if "Overview" in page:
    current_aqi = int(df["aqi"].dropna().iloc[-1])
    color, label, desc = aqi_info(current_aqi)

    if current_aqi > 300:
        st.markdown(f"<div class='alert-haz'>🚨 HAZARDOUS AIR QUALITY ALERT — AQI {current_aqi} — {desc}</div>", unsafe_allow_html=True); st.markdown("")
    elif current_aqi > 200:
        st.markdown(f"<div class='alert-warn'>⚠️ UNHEALTHY AIR — AQI {current_aqi} — {desc}</div>", unsafe_allow_html=True); st.markdown("")
    elif current_aqi > 150:
        st.warning(f"⚠️ AQI {current_aqi} — Unhealthy for Sensitive Groups.")

    col_hero, col_stats = st.columns([1, 2])
    with col_hero:
        st.markdown(f"<div class='aqi-hero'><div class='aqi-number' style='color:{color}'>{current_aqi}</div><div class='aqi-label' style='color:{color}'>{label}</div><div class='aqi-city'>📍 LAHORE, PK · LIVE</div><div style='margin-top:12px;color:rgba(255,255,255,0.5);font-size:0.82rem'>{desc}</div></div>", unsafe_allow_html=True)
    with col_stats:
        avg24 = df.tail(24)["aqi"].mean(); max7d = df.tail(168)["aqi"].max(); min7d = df.tail(168)["aqi"].min()
        c_avg,_,_ = aqi_info(int(avg24)); c_max,_,_ = aqi_info(int(max7d))
        c1,c2,c3,c4 = st.columns(4)
        for col, val, name, clr in [(c1,f"{avg24:.0f}","24h Avg",c_avg),(c2,f"{max7d:.0f}","7d Max",c_max),(c3,f"{min7d:.0f}","7d Min","#4fc3f7"),(c4,f"{len(df):,}","Records","#7c83fd")]:
            col.markdown(f"<div class='metric-card'><div class='metric-val' style='color:{clr}'>{val}</div><div class='metric-name'>{name}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 Last 7 Days")
    df7 = df.tail(168)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df7["timestamp"], y=df7["aqi"], mode="lines", fill="tozeroy",
                              line=dict(color="rgba(79,195,247,0.8)",width=1.5), fillcolor="rgba(79,195,247,0.05)", name="AQI"))
    for lvl,clr,nm in [(50,"#00e676","Good"),(100,"#ffea00","Moderate"),(150,"#ff9100","USG"),(200,"#ff1744","Unhealthy")]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=clr, opacity=0.35, annotation_text=nm, annotation_font_color=clr, annotation_font_size=10)
    fig.update_layout(height=320, **PLOT_CFG); st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🥧 Category Distribution")
        if "aqi_cat_label" in df.columns:
            counts = df["aqi_cat_label"].value_counts()
            fig2 = px.pie(values=counts.values, names=counts.index, hole=0.5)
            fig2.update_layout(height=280, **PLOT_CFG, legend=dict(font=dict(size=11))); st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.markdown("#### 🕐 Hourly Pattern")
        if "hour" in df.columns:
            hourly = df.groupby("hour")["aqi"].mean().reset_index()
            fig3 = px.bar(hourly, x="hour", y="aqi", color="aqi", color_continuous_scale="RdYlGn_r")
            fig3.update_layout(height=280, **PLOT_CFG, coloraxis_showscale=False); st.plotly_chart(fig3, use_container_width=True)

# ── 3-DAY FORECAST ────────────────────────────────────────────────────────────
elif "Forecast" in page:
    st.markdown("## 🔮 3-Day AQI Forecast")
    st.caption(f"Model: **{model_choice}** · Horizon: 72 hours")

    if active_model is None or not feat_cols:
        st.warning("⚠️ No trained model found. Run `python stage4_train_model.py` first.")
    else:
        forecast = make_forecast(df, active_model, imputer, feat_cols, hours=72)
        max_fcast = forecast["aqi"].max()
        c_f,l_f,d_f = aqi_info(int(max_fcast))
        if max_fcast > 200:
            st.markdown(f"<div class='alert-warn'>⚠️ Forecast shows AQI reaching {max_fcast:.0f} ({l_f}) in next 3 days</div>", unsafe_allow_html=True); st.markdown("")

        st.markdown("#### Daily Summary")
        cols = st.columns(3)
        for i,(col,lbl) in enumerate(zip(cols,["Today +1","Tomorrow","Day 3"])):
            day_data = forecast.iloc[i*24:(i+1)*24]
            avg=day_data["aqi"].mean(); mx=day_data["aqi"].max(); mn=day_data["aqi"].min()
            clr,lb,_ = aqi_info(int(avg))
            col.markdown(f"<div class='fcard'><div class='fday'>{lbl}</div><div class='faqi' style='color:{clr}'>{avg:.0f}</div><div class='flabel' style='color:{clr}'>{lb}</div><div style='color:rgba(255,255,255,0.35);font-size:0.78rem;margin-top:8px'>↑ {mx:.0f} · ↓ {mn:.0f}</div></div>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("#### Hourly Forecast — Next 72 Hours")
        hist48 = df.tail(48)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist48["timestamp"], y=hist48["aqi"], mode="lines", name="Historical",
                                  line=dict(color="#4fc3f7",width=2), fill="tozeroy", fillcolor="rgba(79,195,247,0.05)"))
        fig.add_trace(go.Scatter(x=forecast["timestamp"], y=forecast["aqi"], mode="lines", name="Forecast (72h)",
                                  line=dict(color="#ff6b35",width=2.5,dash="dash"), fill="tozeroy", fillcolor="rgba(255,107,53,0.06)"))
        fig.add_trace(go.Scatter(
            x=list(forecast["timestamp"])+list(forecast["timestamp"][::-1]),
            y=list(forecast["aqi"]*1.1)+list(forecast["aqi"]*0.9)[::-1],
            fill="toself", fillcolor="rgba(255,107,53,0.08)", line=dict(color="rgba(0,0,0,0)"), name="Confidence Band"))
        # for d in range(1,4):
        #     fig.add_vline(x=df["timestamp"].iloc[-1]+timedelta(days=d), line_dash="dot",
        #                   line_color="rgba(255,255,255,0.15)", annotation_text=f"Day {d}", annotation_font_color="rgba(255,255,255,0.4)")
        # last_valid_ts = df["timestamp"].dropna().iloc[-1]
        # for d in range(1,4):
        #     fig.add_vline(x=last_valid_ts+timedelta(days=d), line_dash="dot",
        #           line_color="rgba(255,255,255,0.15)", annotation_text=f"Day {d}", annotation_font_color="rgba(255,255,255,0.4)")
        for d in range(1,4):
            idx = d * 24
            if idx < len(forecast):
                sep_ts = str(forecast["timestamp"].iloc[idx])
                fig.add_shape(type="line", x0=sep_ts, x1=sep_ts, y0=0, y1=1,
                      xref="x", yref="paper", line=dict(color="rgba(255,255,255,0.2)", dash="dot", width=1))
        for lvl,clr,_ in [(50,"#00e676","G"),(100,"#ffea00","M"),(150,"#ff9100","U"),(200,"#ff1744","UH")]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=clr, opacity=0.3)
        fig.update_layout(height=420, legend=dict(bgcolor="rgba(0,0,0,0)"), **PLOT_CFG)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Full 72-Hour Forecast Table"):
            forecast["Time"]     = forecast["timestamp"].dt.strftime("%a %b %d  %H:%M")
            forecast["AQI"]      = forecast["aqi"].round(1)
            forecast["Category"] = forecast["aqi"].apply(lambda x: aqi_info(int(x))[1])
            forecast["Status"]   = forecast["aqi"].apply(lambda x: "🟢" if x<=50 else "🟡" if x<=100 else "🟠" if x<=150 else "🔴" if x<=200 else "🟣")
            st.dataframe(forecast[["Time","AQI","Category","Status"]], use_container_width=True, height=400)

# ── HISTORICAL EDA ────────────────────────────────────────────────────────────
elif "Historical" in page:
    st.markdown("## 📈 Historical EDA")
    days = st.slider("Time window (days)", 7, 120, 30)
    df_f = df.tail(days * 24).copy()
    df_f["aqi_7d_avg"] = df_f["aqi"].rolling(window=168, min_periods=1).mean()

    st.markdown("#### AQI Trend with 7-Day Moving Average")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_f["timestamp"], y=df_f["aqi"], name="AQI", mode="lines", line=dict(color="rgba(79,195,247,0.5)",width=1)))
    fig.add_trace(go.Scatter(x=df_f["timestamp"], y=df_f["aqi_7d_avg"], name="7-Day Avg", mode="lines", line=dict(color="#ff6b35",width=2.5)))
    fig.update_layout(height=350, **PLOT_CFG); st.plotly_chart(fig, use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("#### 🗓️ Day-of-Week Pattern")
        if "day_of_week" in df.columns:
            dow = df.groupby("day_of_week")["aqi"].mean().reset_index()
            dow["day"] = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            fig2 = px.bar(dow, x="day", y="aqi", color="aqi", color_continuous_scale="RdYlGn_r")
            fig2.update_layout(height=300, **PLOT_CFG, coloraxis_showscale=False); st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.markdown("#### 🌡️ Hour × Day Heatmap")
        if "hour" in df.columns and "day_of_week" in df.columns:
            pivot = df.pivot_table(values="aqi", index="day_of_week", columns="hour", aggfunc="mean")
            pivot.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            fig3 = px.imshow(pivot, color_continuous_scale="RdYlGn_r", aspect="auto")
            fig3.update_layout(height=300, **PLOT_CFG); st.plotly_chart(fig3, use_container_width=True)

    # st.markdown("#### 📅 Monthly Average AQI")
    # # df["ym"] = df["timestamp"].dt.to_period("M").astype(str)
    # df["ym"] = df["timestamp"].dt.strftime("%Y-%m")
    # monthly = df.groupby("ym")["aqi"].mean().reset_index()
    # fig4 = px.bar(monthly, x="ym", y="aqi", color="aqi", color_continuous_scale="RdYlGn_r", labels={"ym":"Month","aqi":"Avg AQI"})
    # fig4.update_layout(height=320, **PLOT_CFG, coloraxis_showscale=False); st.plotly_chart(fig4, use_container_width=True)
    # st.markdown("#### 📅 Monthly Average AQI")
    # df_m = df.copy()
    # df_m["ym"] = df_m["timestamp"].dt.tz_localize(None).dt.strftime("%Y-%m")
    # monthly = df_m.groupby("ym")["aqi"].mean().reset_index()
    # fig4 = px.bar(monthly, x="ym", y="aqi", color="aqi", #color_continuous_scale="RdYlGn_r", labels={"ym":"Month","aqi":"Avg AQI"})
    # fig4.update_layout(height=320, **PLOT_CFG, coloraxis_showscale=False)
    # st.plotly_chart(fig4, use_container_width=True)
    # st.markdown("#### 📅 Monthly Average AQI")
    # df_m = df.copy()
    # df_m["ym"] = df_m["timestamp"].astype(str).str[:7]
    # monthly = df_m.groupby("ym")["aqi"].mean().reset_index()
    # st.write("Sample ym values:", monthly["ym"].head(10).tolist())
    # fig4 = px.bar(monthly, x="ym", y="aqi", color="aqi", color_continuous_scale="RdYlGn_r", labels={"ym":"Month","aqi":"Avg AQI"})
    # fig4.update_layout(height=320, **PLOT_CFG, coloraxis_showscale=False)
    # st.plotly_chart(fig4, use_container_width=True)
    st.markdown("#### 📅 Monthly Average AQI")
    df_m = df.copy()
    df_m["ym"] = df_m["timestamp"].astype(str).str[:7]
    monthly = df_m.groupby("ym")["aqi"].mean().reset_index()
    monthly.columns = ["Month", "Avg AQI"]
    fig4 = px.bar(monthly, x="Month", y="Avg AQI",
              color="Avg AQI", color_continuous_scale="RdYlGn_r",
              text="Avg AQI")
    fig4.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig4.update_xaxes(type="category")
    fig4.update_layout(height=320, **PLOT_CFG, coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### 🔗 Feature Correlations with AQI")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "aqi" in num_cols:
        corr = df[num_cols].corr()["aqi"].drop("aqi").abs().sort_values(ascending=False).head(20)
        fig5 = px.bar(x=corr.values, y=corr.index, orientation="h", color=corr.values, color_continuous_scale="Blues_r")
        fig5.update_layout(height=400, **PLOT_CFG, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig5, use_container_width=True)

# ── MODEL REGISTRY ────────────────────────────────────────────────────────────
elif "Registry" in page:
    st.markdown("## 🤖 Model Registry")
    metrics_path = Path("aqi_model_outputs/latest_metrics.csv")
    if metrics_path.exists():
        m = pd.read_csv(metrics_path).iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        for col,val,name,clr in [(c1,f"{m.get('r2','N/A'):.4f}","R² Score","#00e676"),(c2,f"{m.get('mae','N/A'):.4f}","MAE","#4fc3f7"),(c3,f"{m.get('rmse','N/A'):.4f}","RMSE","#ff6b35"),(c4,str(int(m.get('train_size',0))),"Train Size","#7c83fd")]:
            col.markdown(f"<div class='metric-card'><div class='metric-val' style='color:{clr}'>{val}</div><div class='metric-name'>{name}</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    registry = load_registry()
    if not registry.empty:
        if "metrics" in registry.columns:
            registry["R²"]   = registry["metrics"].apply(lambda x: round(x.get("r2",0),4)   if isinstance(x,dict) else "—")
            registry["MAE"]  = registry["metrics"].apply(lambda x: round(x.get("mae",0),4)  if isinstance(x,dict) else "—")
            registry["RMSE"] = registry["metrics"].apply(lambda x: round(x.get("rmse",0),4) if isinstance(x,dict) else "—")
        show = [c for c in ["run_id","model_name","status","trained_at","R²","MAE","RMSE"] if c in registry.columns]
        st.dataframe(registry[show], use_container_width=True, height=350)
        if "R²" in registry.columns:
            st.markdown("#### 📈 R² Over Runs")
            fig = px.line(registry[::-1].reset_index(drop=True), y="R²", markers=True, color_discrete_sequence=["#4fc3f7"])
            fig.update_layout(height=300, **PLOT_CFG); st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No models registered yet. Run `python stage4_train_model.py`")

# ── SHAP EXPLAINER ────────────────────────────────────────────────────────────
elif "SHAP" in page:
    st.markdown("## 🔍 SHAP Feature Importance Explainer")
    if active_model is None:
        st.warning("No model loaded.")
    else:
        try:
            import shap
            sample = df.tail(200).copy()
            sample["hour"]        = sample["timestamp"].dt.hour
            sample["day_of_week"] = sample["timestamp"].dt.dayofweek
            sample["month"]       = sample["timestamp"].dt.month
            sample["day_of_year"] = sample["timestamp"].dt.dayofyear
            sample["is_weekend"]  = (sample["day_of_week"] >= 5).astype(int)
            sample["is_night"]    = ((sample["hour"] >= 22) | (sample["hour"] <= 6)).astype(int)
            X_s = prepare_X(sample, feat_cols)
            X_imp = pd.DataFrame(imputer.transform(X_s), columns=feat_cols) if imputer else X_s
            with st.spinner("Computing SHAP values..."):
                explainer = shap.TreeExplainer(active_model)
                shap_vals = explainer.shap_values(X_imp)
                mean_shap = pd.DataFrame(np.abs(shap_vals), columns=feat_cols).mean().sort_values(ascending=False).head(20)
            st.markdown("#### Global Feature Importance (SHAP)")
            fig = px.bar(x=mean_shap.values, y=mean_shap.index, orientation="h", color=mean_shap.values, color_continuous_scale="Viridis_r")
            fig.update_layout(height=500, **PLOT_CFG, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### Top 3 Features")
            top3 = mean_shap.head(3)
            cols = st.columns(3)
            for col,(feat,val),clr in zip(cols,top3.items(),["#00e676","#4fc3f7","#ff6b35"]):
                col.markdown(f"<div class='metric-card'><div class='metric-val' style='color:{clr}'>{val:.3f}</div><div class='metric-name'>{feat}</div></div>", unsafe_allow_html=True)
        except ImportError:
            st.warning("Install SHAP: `pip install shap`")
        except Exception as e:
            st.error(f"SHAP error: {e}")
            if hasattr(active_model, "feature_importances_"):
                imp = pd.Series(active_model.feature_importances_, index=feat_cols).sort_values(ascending=False).head(20)
                fig = px.bar(x=imp.values, y=imp.index, orientation="h", color=imp.values, color_continuous_scale="Viridis_r")
                fig.update_layout(height=500, **PLOT_CFG, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)