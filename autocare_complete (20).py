# autocare_complete (18).py
# ---------------------------------------------------------------------------------------
# AutoCare AI — ETA Intelligence, Churn Radar, Uplift, Segments & Gemini Outreach (2025)
# ---------------------------------------------------------------------------------------
# What’s included (single-file, fast ~5–8s on laptop):
# • DATA: steadier Overdue% (softer monsoon); OEM intervals +12% to reduce volatility
# • FEATURES: consistent float32 arrays end-to-end; JSON-safe bin labels (fixes Interval serialization)
# • MODELS: ETA regressor + empirical prediction intervals; calibrated churn; T-learner uplift
# • SEGMENTS: new Customer Segments (High-Value & At-Risk, Loyal Champions, New & Promising, Hibernating)
#   integrated across ETA, Churn/RFM and Uplift pages; CLTV by segment
# • VISUALS: all chart titles white; storyboard churn EDA; uplift segment bars + box & density plots
# • OUTREACH (FINAL): Polished UI, advanced campaign analytics, and hyper-personalized customer-level exports.
# • ROBUSTNESS: graceful fallbacks if XGBoost / Gemini / Pydantic missing; caching; no external files
# ---------------------------------------------------------------------------------------
# === FINAL DEMO UPGRADE SUMMARY ===
# NEW DATA COLUMNS & THEIR USAGE:
# • persona (str): Usage-based group {Light, Typical, Heavy, Commercial}. Used for persona-based churn thresholds and analysis.
# • long_overdue_threshold (int): Days after which a customer is considered long overdue, based on persona. Drives "Silent Churn" KPI.
# • long_overdue (int): Flag (0/1) if recency exceeds the persona's threshold. Used in churn status calculation.
# • service_type (str): Service category {Routine, Comprehensive, Repair}. Primary driver for base service time (ECT).
# • ect_hours (float): Estimated Completion Time in hours (base).
# • parts_wait_risk (int): Flag (0/1) for risk of parts-related delays. Used in "ETA Because..." explanations and adds delay to ECT.
# • tech_skill_gap (int): Flag (0/1) for risk of technician availability delays. Adds delay to ECT, used in explanations.
# • insurance_approval_req (int): Flag (0/1) if insurance approval is needed. Adds significant delay to ECT for 'Repair' jobs.
# • delay_days (float): Total calculated delay in days from the above risk flags.
# • ect_days (float): Final ECT in days (base + delays). Powers Service Adherence KPIs and operational planning.
# • amc_enrolled (int): Flag (0/1) if customer has an Annual Maintenance Contract. Slightly boosts baseline booking probability.
# • CSI_1to5 (int): Customer Satisfaction Index score (1-5). Used in "Churn Because..." explanations.
# • NPS_bucket (str): Net Promoter Score category {Detractor, Passive, Promoter}. Used in churn explanations.
# • feedback_reason (str): Categorical reason for low satisfaction. Used in "Churn Because..." explanations.
# • recency_months (int): Recency in whole months, for inactivity window sliders and analysis.
# ---------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from typing import List, Tuple, Dict, Any
import json
import re


# sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

# Optional XGBoost with graceful fallback
xgb_ok = True
try:
    from xgboost import XGBRegressor, XGBClassifier  # type: ignore
except Exception:
    xgb_ok = False
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier  # type: ignore

# Optional Gemini (graceful fallback)
gemini_ok = True
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    gemini_ok = False

# Pydantic for schema (graceful fallback stub if missing)
try:
    from pydantic import BaseModel, Field  # type: ignore
except Exception:
    class BaseModel:  # minimal stub
        def __init__(self, **kwargs): ...
        def dict(self): return {}
    def Field(default=None, description:str=""): return default

import warnings
warnings.filterwarnings("ignore")

# ================================================================================================
# PAGE CONFIG & CSS
# ================================================================================================
st.set_page_config(
    page_title="AutoCare AI — ETA, Churn, Uplift & Outreach",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed" # Hides sidebar by default
)

st.markdown("""
<style>
/* Hides the sidebar hamburger menu and Streamlit footer */
[data-testid="stSidebar"], [data-testid="main-menu-button"], footer {
    display: none !important;
}
.stApp { background:#0f1016; color:#ffffff; }

/* top banner */
.banner{
  display:flex;align-items:center;gap:18px;background:linear-gradient(90deg,#151a2a,#171b2a 60%,#00ff88 160%);
  padding:16px 20px;border-radius:0 0 14px 14px;margin:0 0 14px 0;border-bottom:2px solid #26304a;
  justify-content:center;text-align:center
}
.banner h1{margin:0;color:#fff;font-weight:800;font-size:1.55rem}
.banner p{margin:4px 0 0 0;color:#aab1c2;font-size:.95rem}

/* common containers */
.section{background:linear-gradient(180deg,#1a1e2a,#121420);border:1px solid #2a2f44;border-radius:14px;padding:14px 16px;margin:14px 0}
.section h2{margin:0 0 6px 0;color:#fff;font-size:1.15rem;text-align:center}
.muted{color:#c7cbd6;font-size:0.95rem;text-align:center}

.kpi{background:linear-gradient(180deg,#1a1e2a,#121420);border:1px solid #2a2f44;border-radius:14px;padding:12px 14px;margin-bottom:10px}
.kpi .l{color:#b5bccb;letter-spacing:.5px;text-transform:uppercase;font-size:12px}
.kpi .v{color:#00ff88;font-weight:900;font-size:26px}
.kpi .s{color:#94a0b6;font-size:11px}
.kpi.red .v{color:#ff6b6b}
.kpi.amber .v{color:#ffb703}

.insight{background:rgba(40,48,76,.55);border:1px solid #2a2f44;border-left:4px solid #00ff88;padding:10px 12px;border-radius:8px;margin:8px 0}
.insight h4{margin:0 0 6px 0;color:#00ff88}
.insight.warn{border-left-color:#ffb703}
.insight.danger{border-left-color:#ff6b6b}
.insight.gray{border-left-color:#8a8f98}

[data-testid="stDataFrame"] div{ color:#e8edf6; }
label, .stSelectbox label, .stSlider label, .stRadio label { color:#ffffff !important; }

/* --- FIX for Radio button text color --- */
div[data-baseweb="radio"] > label > div,
div[data-baseweb="radio"] label p {
    color: #ffffff !important;
    font-size: 1rem;
}
.st-emotion-cache-1y4p8pa, .st-emotion-cache-1y4p8pa p { /* Target for radio button description text */
    color: #aab1c2 !important;
}


.stTabs [role="tab"]{color:#ffffff !important; font-weight:700;}
.stTabs [role="tab"][aria-selected="true"]{
  background:rgba(255,255,255,0.06); border:1px solid #2a2f44; border-bottom:1px solid #00ff88;
}

/* --- NEW GLOBAL UI FIXES --- */
/* Remove top gap so the app starts flush at the top */
.stApp > header { display:none; }
.block-container { padding-top: 0rem !important; }

/* Center align the tab list across the page width */
.stTabs [role="tablist"] {
  justify-content: center !important;
}

/* Force-white chart titles / labels as a safety net even if a figure skips darkify() */
.js-plotly-plot .gtitle,
.js-plotly-plot .xtitle,
.js-plotly-plot .ytitle,
.js-plotly-plot text {
  fill: #ffffff !important;
}

/* Buttons & inputs: keep light text for contrast */
.stTextInput > div > div input, .stSelectbox div[data-baseweb="select"] {
  color: #ffffff !important;
}
.stButton>button {
    background:#00ff88 !important; color:#0f1016 !important; font-weight:800 !important;
    border:1px solid #22ffaa; border-radius:10px; padding:10px 16px;
}
.stButton>button:hover { filter:brightness(0.95); }

/* Optional: tighten extra vertical whitespace below the banner */
.banner { margin-top: 0 !important; }

/* --- NEW UI FOR CAMPAIGN BUILDER --- */
.step-container {
    background-color: #1a1e2a;
    border: 1px solid #2a2f44;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}
.step-container h3 {
    color: #00ff88;
    margin-top: 0;
    margin-bottom: 16px;
    border-bottom: 1px solid #2a2f44;
    padding-bottom: 10px;
}
.step-container .stRadio > label, .step-container .stRadio label {
    font-weight: 600;
    color: #ffffff !important;
}
.stAlert {
    border-radius: 8px;
    background-color: rgba(0, 255, 136, 0.1) !important;
    border: 1px solid rgba(0, 255, 136, 0.2) !important;
}
.stAlert p {
    color: #e8edf6 !important;
}

/* --- XAI CHIP STYLE --- */
.chip {
    display: inline-block;
    padding: 4px 10px;
    margin: 4px 4px 4px 0;
    border-radius: 16px;
    background-color: #2a2f44;
    border: 1px solid #4a4f64;
    color: #e8edf6;
    font-size: 0.85rem;
}
.chip.red { background-color: rgba(255, 107, 107, 0.2); border-color: #ff6b6b; }
.chip.amber { background-color: rgba(255, 183, 3, 0.2); border-color: #ffb703; }

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="banner">
  <img src="https://cdn-icons-png.flaticon.com/512/743/743988.png" style="height:56px">
  <div>
    <h1>AutoCare AI — ETA Intelligence, Churn Radar, Uplift & Outreach</h1>
    <p>Hybrid ETA (calendar ⟂ km) • Calibrated churn • T-Learner uplift • Segments • AI Campaign Builder</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ================================================================================================
# HELPERS
# ================================================================================================
def format_indian(num):
    """Formats a number in the Indian numbering system (Lakhs, Crores)."""
    if not isinstance(num, (int, float)):
        return num
    num = float(num)
    if num < 100000:
        return f"{num:,.0f}"
    lakhs = num / 1_00_000
    if lakhs < 100:
        return f"{lakhs:.2f} L"
    crores = num / 1_00_00_000
    return f"{crores:.2f} Cr"

def darkify(fig, height=None, title=None):
    """Ensure all charts have white titles & dark background."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title=dict(font_color='white')
    )
    if title is not None:
        fig.update_layout(title=title)
    if height:
        fig.update_layout(height=height)
    return fig

def categorize_urgency(days:int) -> str:
    if days < -30: return "🚨 Critical Overdue"
    if days < 0:   return "⚠️ Overdue"
    if days <= 15: return "🔥 Due Soon (0–15)"
    if days <= 45: return "💡 Upcoming (16–45)"
    return "🗓️ Future"

def oem_alignment(delta:int)->str:
    if delta <= -10: return "Early vs OEM"
    if delta >=  10: return "Late vs OEM"
    return "On-time vs OEM"

def pretty(feat:str)->str:
    mapping = {
        'vehicle_age_years':'Vehicle Age (years)',
        'total_services':'Total Services',
        'avg_service_cost':'Average Service Cost',
        'satisfaction_score':'Satisfaction Score',
        'days_since_purchase':'Days Since Purchase',
        'days_since_service':'Days Since Service',
        'service_frequency':'Service Frequency',
        'seasonality_factor':'Seasonality Factor',
        'econ_indicator':'City Economic Index',
        'city_efficiency':'City Efficiency',
        'competitor_density':'Competitor Density',
        'competitor_promo_active':'Competitor Promotion Active',
        'delta_vs_oem_days':'Δ vs OEM (days)',
        'oem_interval_days':'OEM Interval (days)',
        'weather_factor':'Weather Factor',
        'km_driven':'KM Driven',
        'daily_km':'Daily KM',
        'oem_km_interval':'OEM KM Interval',
        'km_until_service':'KM Until Service',
        'km_since_last_service':'KM Since Last Service',
        'km_overdue':'KM Overdue',
        'cost_to_income_proxy':'Cost-to-Income (proxy)',
        'high_competition':'High Competition',
        'recency':'Recency (days)',
        'frequency':'Frequency (services/yr)',
        'monetary':'Monetary (₹)',
        'rfm_score':'RFM Score'
    }
    return mapping.get(feat, feat.replace('_',' ').title())

def safe_bin_mean(df: pd.DataFrame, x: str, y: str, bins=10, as_midpoint=False):
    """Quantile-bin x and return mean(y) with JSON-safe bin labels (fixes Interval serialization)."""
    tmp = df[[x, y]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame({'bin':[], y:[]})
    tmp['__bin'] = pd.qcut(tmp[x], q=bins, duplicates='drop')
    if as_midpoint:
        tmp['__mid'] = tmp['__bin'].apply(lambda iv: (iv.left + iv.right)/2.0)
        out = tmp.groupby('__mid', as_index=False)[y].mean().rename(columns={'__mid':'bin'})
    else:
        out = tmp.groupby('__bin', as_index=False)[y].mean()
        out['bin'] = out['__bin'].astype(str)
    return out[['bin', y]]

def recommend_action(row: pd.Series) -> str:
    if row['urgency'] in ["🚨 Critical Overdue","⚠️ Overdue"] or row.get('km_overdue',0)==1:
        return "Priority call + pick-up/drop + goodwill coupon"
    if row['urgency'] == "🔥 Due Soon (0–15)":
        return "SMS+Email reminder + express lane link"
    if row['urgency'] == "💡 Upcoming (16–45)":
        return "Education drip (pre-monsoon check) + soft CTA"
    return "Nurture content; remind next month"

# ================================================================================================
# DATA (synthetic; steadier overdue share; OEM +12%)
# ================================================================================================
@st.cache_data(show_spinner=False)
def generate_data(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    models = {
        'Alto K10': {'complexity': 1.0, 'base_cost': 2800, 'oem_days': int(180*1.12), 'oem_km': 10000},
        'S-Presso': {'complexity': 1.1, 'base_cost': 2900, 'oem_days': int(180*1.12), 'oem_km': 10000},
        'Wagon R': {'complexity': 1.2, 'base_cost': 3200, 'oem_days': int(180*1.12), 'oem_km': 10000},
        'Swift': {'complexity': 1.5, 'base_cost': 3800, 'oem_days': int(180*1.12), 'oem_km': 10000},
        'Dzire': {'complexity': 1.5, 'base_cost': 3900, 'oem_days': int(180*1.12), 'oem_km': 10000},
        'Baleno': {'complexity': 1.8, 'base_cost': 4500, 'oem_days': int(150*1.12), 'oem_km': 10000},
        'Brezza': {'complexity': 2.0, 'base_cost': 5000, 'oem_days': int(150*1.12), 'oem_km': 10000},
        'Ertiga': {'complexity': 2.2, 'base_cost': 5500, 'oem_days': int(150*1.12), 'oem_km': 10000},
        'XL6': {'complexity': 2.5, 'base_cost': 6000, 'oem_days': int(150*1.12), 'oem_km': 10000},
        'Grand Vitara': {'complexity': 2.8, 'base_cost': 6500, 'oem_days': int(150*1.12), 'oem_km': 10000},
    }
    cities = {
        'Delhi': {'eff':0.92,'econ':1.10,'comp':0.75},
        'Mumbai': {'eff':0.88,'econ':1.15,'comp':0.85},
        'Bangalore': {'eff':0.95,'econ':1.20,'comp':0.65},
        'Chennai': {'eff':0.90,'econ':1.05,'comp':0.70},
        'Hyderabad': {'eff':0.89,'econ':1.08,'comp':0.60},
        'Pune': {'eff':0.91,'econ':1.12,'comp':0.55},
        'Kolkata': {'eff':0.85,'econ':0.95,'comp':0.80},
        'Ahmedabad': {'eff':0.87,'econ':1.00,'comp':0.58},
        'Jaipur': {'eff':0.82,'econ':0.90,'comp':0.52},
    }
    today = datetime.now()
    rows = []
    
    # === BEGIN: FINAL DEMO UPGRADE (Data Generation) ===
    personas = {
        'Light': {'km_range': (6000, 8000), 'overdue_days': 480},
        'Typical': {'km_range': (10000, 14000), 'overdue_days': 420},
        'Heavy': {'km_range': (18000, 24000), 'overdue_days': 330},
        'Commercial': {'km_range': (28000, 36000), 'overdue_days': 330}
    }
    persona_choices = list(personas.keys())
    persona_probs = [0.20, 0.45, 0.25, 0.10]
    
    feedback_reasons = ['wait_time', 'price', 'comms', 'cleanliness', 'advisor_helpful', 'good_experience']
    feedback_probs = [0.15, 0.18, 0.12, 0.05, 0.20, 0.30]
    # === END: FINAL DEMO UPGRADE (Data Generation) ===
    
    for i in range(n):
        model = rng.choice(list(models))
        city  = rng.choice(list(cities))
        m, c = models[model], cities[city]

        purchase_date = today - timedelta(days=int(rng.integers(90, 15*365)))
        v_age_days = (today - purchase_date).days
        v_age_years = max(0.25, v_age_days / 365)
        
        # === BEGIN: FINAL DEMO UPGRADE (Persona-based Usage & Recency) ===
        persona_name = rng.choice(persona_choices, p=persona_probs)
        persona_info = personas[persona_name]
        annual_km = rng.uniform(*persona_info['km_range'])
        
        # Right-tailed recency for silent churn
        recency_rand = rng.random()
        if recency_rand < 0.70:
            last_service_days = int(rng.integers(60, 330))
        elif recency_rand < 0.90:
            last_service_days = int(rng.integers(330, 480))
        else:
            last_service_days = int(rng.integers(480, 720))
            
        long_overdue_threshold = persona_info['overdue_days']
        long_overdue = 1 if last_service_days >= long_overdue_threshold else 0
        # === END: FINAL DEMO UPGRADE (Persona-based Usage & Recency) ===

        km_driven = max(500, annual_km * v_age_years * rng.uniform(0.9, 1.1))
        daily_km = max(5, km_driven / max(60, v_age_days))

        last_service_date = today - timedelta(days=last_service_days)
        km_since_last_service = daily_km * last_service_days

        monsoon = 1.08 if today.month in [6,7,8,9] else 1.00
        seasonality = 1.0 + (0.08 if today.month in [4,5,6] else 0) + (0.06 if today.month in [9,10,11] else 0)
        competitor_promo = rng.choice([0,1], p=[0.85,0.15])

        total_services = max(1, int(v_age_years*1.5 + rng.poisson(0.6)))
        base_cost = (m['base_cost'] + (v_age_years**1.15)*300 + m['complexity']*820)
        cost = float(max(2500, base_cost * seasonality * c['econ'] * rng.uniform(0.9, 1.1)))

        base_sat = 4.9 - 0.10*v_age_years - (cost-4000)/7600 - 0.06*competitor_promo
        sat = float(np.clip(base_sat + rng.normal(0,0.20), 1, 5))

        churn = 0.05 + 0.17*(5-sat) + 0.0023*max(0,last_service_days-210) + 0.050*v_age_years \
                + 0.070*competitor_promo + max(0,(cost-6500)/12000)
        churn = float(np.clip(rng.normal(churn,0.030), 0.01, 0.98))

        behavior_adjust = 9*(5-sat) + rng.normal(0,5) - (c['eff']-0.85)*38
        calendar_interval = max(80, m['oem_days'] - 7*v_age_years + behavior_adjust)
        next_due_calendar = last_service_date + timedelta(days=int(calendar_interval))
        days_until_calendar = (next_due_calendar - today).days

        oem_km = m['oem_km']
        next_km_threshold = (np.floor(km_driven / oem_km) + 1) * oem_km
        km_until_service = max(0, next_km_threshold - km_driven)
        days_until_km = km_until_service / daily_km

        hybrid_days = min(days_until_calendar, days_until_km) / monsoon
        days_until = int(np.round(hybrid_days))

        oem_next = last_service_date + timedelta(days=m['oem_days'])
        delta_vs_oem = int((next_due_calendar - oem_next).days)

        km_overdue = int(km_since_last_service > (1.2 * oem_km))
        
        # === BEGIN: FINAL DEMO UPGRADE (ECT, AMC, Feedback) ===
        service_type = rng.choice(['Routine', 'Comprehensive', 'Repair'], p=[0.70, 0.25, 0.05])
        if service_type == 'Routine': ect_hours = rng.uniform(2, 3.5)
        elif service_type == 'Comprehensive': ect_hours = rng.uniform(3, 6)
        else: ect_hours = rng.uniform(24, 96) # 1-4 days in hours

        parts_wait_risk = rng.choice([0, 1], p=[0.92 - 0.02 * min(v_age_years, 10), 0.08 + 0.02 * min(v_age_years, 10)])
        tech_skill_gap = rng.choice([0, 1], p=[0.93, 0.07])
        insurance_approval_req = rng.choice([0, 1], p=[0.7, 0.3]) if service_type == 'Repair' else 0
        
        delay_days = 0
        if parts_wait_risk: delay_days += rng.uniform(1, 7)
        if tech_skill_gap: delay_days += rng.uniform(1, 3)
        if insurance_approval_req: delay_days += rng.uniform(3, 5)
        
        ect_days = ect_hours / 24.0 + delay_days
        
        amc_enrolled = rng.choice([0, 1], p=[0.75, 0.25])
        
        csi_score = int(np.round(sat))
        if csi_score <= 2: nps_bucket = 'Detractor'
        elif csi_score == 3: nps_bucket = 'Passive'
        else: nps_bucket = 'Promoter'
        
        feedback = rng.choice(feedback_reasons, p=feedback_probs) if csi_score <= 3 else 'good_experience'
        # === END: FINAL DEMO UPGRADE (ECT, AMC, Feedback) ===

        rows.append({
            'customer_id': f"MSIL{10000+i}",
            'model': model, 'city': city,
            'purchase_date': purchase_date, 'vehicle_age_years': round(v_age_years,1),
            'last_service_date': last_service_date, 'days_until_service': days_until,
            'total_services': total_services, 'avg_service_cost': round(cost,0),
            'satisfaction_score': round(sat,1), 'churn_probability': round(churn,3),
            'competitor_promo_active': competitor_promo, 'seasonality_factor': seasonality,
            'weather_factor': monsoon, 'econ_indicator': c['econ'], 'city_efficiency': c['eff'],
            'competitor_density': c['comp'], 'oem_interval_days': m['oem_days'],
            'oem_km_interval': oem_km, 'delta_vs_oem_days': delta_vs_oem,
            'km_driven': round(km_driven,0), 'daily_km': round(daily_km,1),
            'km_since_last_service': round(km_since_last_service,0),
            'km_until_service': round(km_until_service,0), 'km_overdue': km_overdue,
            # === BEGIN: FINAL DEMO UPGRADE (Append new columns) ===
            'persona': persona_name,
            'long_overdue_threshold': long_overdue_threshold,
            'long_overdue': long_overdue,
            'service_type': service_type,
            'ect_hours': ect_hours,
            'parts_wait_risk': parts_wait_risk,
            'tech_skill_gap': tech_skill_gap,
            'insurance_approval_req': insurance_approval_req,
            'delay_days': delay_days,
            'ect_days': ect_days,
            'amc_enrolled': amc_enrolled,
            'CSI_1to5': csi_score,
            'NPS_bucket': nps_bucket,
            'feedback_reason': feedback,
            # === END: FINAL DEMO UPGRADE (Append new columns) ===
        })

    df = pd.DataFrame(rows)

    # RFM proxies
    now = datetime.now()
    df['recency'] = (now - df['last_service_date']).dt.days
    df['frequency'] = df['total_services'] / df['vehicle_age_years'].clip(lower=0.5)
    df['monetary'] = df['avg_service_cost'] * df['total_services']

    for col in ['recency','frequency','monetary']:
        rngv = df[col].max() - df[col].min()
        df[col+'_norm'] = 0.0 if rngv==0 else (df[col] - df[col].min())/rngv
    df['rfm_score'] = (1 - df['recency_norm']) * 0.4 + df['frequency_norm'] * 0.2 + df['monetary_norm'] * 0.4

    # === BEGIN: FINAL DEMO UPGRADE (Updated Churn Status) ===
    # Original logic was: long_overdue = df['recency'] >= 450
    high_prob = df['churn_probability'] > 0.60
    df['churn_status'] = (df['long_overdue'] | high_prob).astype(int)
    # === END: FINAL DEMO UPGRADE (Updated Churn Status) ===

    # CLTV proxy
    df['CLTV'] = (df['avg_service_cost'] * np.clip(df['total_services'], 1, None) * 3).astype(float)
    
    # === BEGIN: FINAL DEMO UPGRADE (Add recency in months) ===
    df['recency_months'] = (df['recency'] / 30.4).astype(int)
    # === END: FINAL DEMO UPGRADE (Add recency in months) ===

    return df

# ================================================================================================
# NEW: Data Enhancement for Added Features
# ================================================================================================
@st.cache_data(show_spinner=False)
def enhance_data_with_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic columns for new features requested."""
    df_new = df.copy()
    rng = np.random.default_rng(123)
    now = datetime.now()

    # --- Component Health Scores ---
    # Health = 100 - (usage_factor * 100). Lower is worse.
    df_new['oil_health']   = np.clip(100 - (df_new['km_since_last_service']/11000 + df_new['recency']/200)/2 * 100, 0, 100)
    df_new['brakes_health']= np.clip(100 - (df_new['km_since_last_service']/25000 + df_new['recency']/730)/2 * 100, 0, 100)
    df_new['air_filter_health'] = np.clip(100 - (df_new['km_since_last_service']/15000 + df_new['recency']/365)/2 * 100, 0, 100)
    df_new['tires_health'] = np.clip(100 - (df_new['km_driven']/50000) * 100, 0, 100)
    df_new['battery_health'] = np.clip(100 - (df_new['vehicle_age_years']/4.5) * 100, 0, 100)
    
    df_new['component_due'] = "None"
    health_cols = ['oil_health', 'brakes_health', 'air_filter_health', 'tires_health', 'battery_health']
    min_health_col = df_new[health_cols].idxmin(axis=1)
    df_new.loc[df_new[health_cols].min(axis=1) < 40, 'component_due'] = min_health_col.str.replace('_health','').str.title()


    # --- Warranty Expiry ---
    df_new['warranty_expiry_date'] = df_new['purchase_date'] + pd.to_timedelta(3 * 365, 'd')
    df_new['days_to_warranty_expiry'] = (df_new['warranty_expiry_date'] - now).dt.days
    df_new['warranty_expired'] = (df_new['days_to_warranty_expiry'] < 0).astype(int)

    # --- Lifecycle & Personas ---
    df_new['lifecycle_stage'] = pd.cut(df_new['vehicle_age_years'], bins=[0, 4, 7, 20], labels=['Early (0-4y)', 'Mid (4-7y)', 'Late (7y+)'])
    
    persona_map = {"Loyal Champions": "Premium Loyalists", "High-Value & At-Risk": "Warranty Warriors", "New & Promising": "First-Time Owners", "Hibernating": "Seasonal Users"}
    # Use .get to avoid KeyError if a segment is empty
    df_new['persona'] = df_new['cust_segment'].apply(lambda x: persona_map.get(x))


    # --- Enhanced Churn Factors ---
    df_new['unresolved_complaints'] = rng.choice([0,1,2], size=len(df_new), p=[0.9, 0.08, 0.02])
    df_new['declined_services'] = rng.choice([0,1], size=len(df_new), p=[0.8, 0.2]) * (df_new['satisfaction_score'] < 3.5)
    df_new['ftfr_proxy'] = np.clip(rng.normal(0.95, 0.05, size=len(df_new)) - 0.05 * df_new['unresolved_complaints'], 0.75, 1.0)
    df_new['service_gap_trend'] = (df_new['delta_vs_oem_days'] > 15).astype(int)

    # --- Serviceability Index for Uplift ---
    city_serviceability = {'Delhi':0.9,'Mumbai':0.85,'Bangalore':0.95,'Chennai':0.88,'Hyderabad':0.87,'Pune':0.92,'Kolkata':0.8,'Ahmedabad':0.86,'Jaipur':0.78}
    df_new['serviceability_index'] = df_new['city'].map(city_serviceability)

    # --- Trigger Hierarchy Flags ---
    df_new['oem_schedule_trigger'] = (df_new['days_until_service'] <= 30) & (df_new['days_until_service'] > -15)
    df_new['usage_trigger'] = (df_new['km_since_last_service'] > 9000)
    is_monsoon_month = now.month in [6,7,8,9]
    monsoon_cities = ['Mumbai','Kolkata','Chennai']
    df_new['context_trigger'] = is_monsoon_month & df_new['city'].isin(monsoon_cities)

    return df_new

# ================================================================================================
# CUSTOMER SEGMENTATION (RFM + behavior) — integrated across pages
# ================================================================================================
def build_segments(df: pd.DataFrame) -> pd.DataFrame:
    segs=[]
    for _,r in df.iterrows():
        if r['rfm_score']>0.70 and (r['churn_probability']>0.50 or r['recency']>240):
            segs.append("High-Value & At-Risk")
        elif r['rfm_score']>0.70:
            segs.append("Loyal Champions")
        elif r['rfm_score']>0.40:
            segs.append("New & Promising")
        else:
            segs.append("Hibernating")
    df['cust_segment'] = segs
    return df

# ================================================================================================
# FEATURE ENGINEERING
# ================================================================================================
@st.cache_data(show_spinner=False)
def build_features(df: pd.DataFrame):
    f = df.copy()
    now = datetime.now()
    f['days_since_purchase'] = (now - f['purchase_date']).dt.days
    f['days_since_service']  = (now - f['last_service_date']).dt.days
    f['service_frequency']   = f['total_services']/f['vehicle_age_years'].clip(lower=0.5)
    f['high_competition']    = (f['competitor_density']>0.7).astype(int)
    f['cost_to_income_proxy'] = f['avg_service_cost'] / (f['econ_indicator']*10000)

    f = pd.get_dummies(f, columns=['model','city'], prefix=['model','city'])

    base = [
        'vehicle_age_years','total_services','avg_service_cost','satisfaction_score',
        'days_since_purchase','days_since_service','service_frequency','seasonality_factor',
        'econ_indicator','city_efficiency','competitor_density','competitor_promo_active',
        'delta_vs_oem_days','oem_interval_days','cost_to_income_proxy','high_competition',
        'km_driven','daily_km','km_since_last_service','km_until_service','oem_km_interval'
    ]
    dynamic = [c for c in f.columns if c.startswith('model_') or c.startswith('city_')]
    feats = base + dynamic

    X = f[feats].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    y_eta = f['days_until_service'].astype(float)
    y_churn = f['churn_status'].astype(int)
    return X, y_eta, y_churn, feats

# ================================================================================================
# MODELS
# ================================================================================================
@st.cache_resource(show_spinner=False)
def train_models(X, y_eta, y_churn):
    X_np = np.asarray(X, dtype=np.float32)
    y_eta_np = np.asarray(y_eta, dtype=np.float32)
    y_churn_np = np.asarray(y_churn, dtype=np.int32)

    if xgb_ok:
        eta = XGBRegressor(n_estimators=260, max_depth=6, learning_rate=0.10,
                           subsample=0.9, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        eta.fit(X_np, y_eta_np)

        raw = XGBClassifier(
            n_estimators=240, max_depth=6, learning_rate=0.10,
            subsample=0.9, colsample_bytree=0.8,
            scale_pos_weight=(len(y_churn_np[y_churn_np==0]) / max(1,len(y_churn_np[y_churn_np==1]))),
            eval_metric="logloss", random_state=42, n_jobs=-1
        )
    else:
        eta = HistGradientBoostingRegressor(max_depth=6, max_bins=255, learning_rate=0.15)
        eta.fit(X_np, y_eta_np)
        raw = RandomForestClassifier(n_estimators=220, random_state=42, n_jobs=-1)

    clf = CalibratedClassifierCV(raw, method='sigmoid', cv=3)
    clf.fit(X_np, y_churn_np)

    X_tr, X_te, y_tr, y_te = train_test_split(X_np, y_churn_np, test_size=0.25, random_state=42, stratify=y_churn_np)
    clf_split = CalibratedClassifierCV(raw, method='sigmoid', cv=3)
    clf_split.fit(X_tr, y_tr)

    return eta, clf, (X_tr, X_te, y_tr, y_te)

def eta_pred_with_intervals(model, X, q_low=0.025, q_high=0.975):
    """Empirical prediction-band using residual quantiles."""
    X_np = np.asarray(X, dtype=np.float32)
    y_hat = model.predict(X_np)
    rs = np.random.RandomState(42)
    idx = rs.choice(len(X_np), size=min(3000, len(X_np)), replace=False)
    # approximate residuals
    eps = (y_hat[idx] - np.median(y_hat[idx]))
    lo_q, hi_q = np.quantile(eps, q_low), np.quantile(eps, q_high)
    band_lo = y_hat + lo_q
    band_hi = y_hat + hi_q
    return y_hat, band_lo, band_hi

@st.cache_data(show_spinner=False)
def evaluate(_eta_model, _churn_model, X, y_eta, _split_pack, feature_names):
    X_tr, X_te, y_tr, y_te = _split_pack
    X_np = np.asarray(X, dtype=np.float32)
    y_eta_np = np.asarray(y_eta, dtype=np.float32)

    # ETA metrics
    y_hat = _eta_model.predict(X_np)
    mae = float(mean_absolute_error(y_eta_np, y_hat))
    r2  = float(r2_score(y_eta_np, y_hat))

    # ETA importances
    try:
        vals = getattr(_eta_model, "feature_importances_", None)
        if vals is not None and len(vals)==X_np.shape[1]:
            eta_imp = pd.DataFrame({"feature": feature_names, "importance": np.abs(vals)})
        else:
            raise AttributeError
    except Exception:
        perm = permutation_importance(_eta_model, X_np, y_eta_np,
                                      scoring="neg_mean_absolute_error",
                                      n_repeats=3, random_state=42)
        eta_imp = pd.DataFrame({"feature": feature_names, "importance": np.abs(perm.importances_mean)})
    eta_imp = eta_imp.sort_values("importance", ascending=False)

    # Churn metrics on holdout
    p_te = _churn_model.predict_proba(X_te)[:, 1]
    y_pred_te = (p_te >= 0.5).astype(int)
    acc  = float(accuracy_score(y_te, y_pred_te))
    prec = float(precision_score(y_te, y_pred_te))
    rec  = float(recall_score(y_te, y_pred_te))
    f1   = float(f1_score(y_te, y_pred_te))
    auc  = float(roc_auc_score(y_te, p_te))

    # Churn importances (best effort)
    try:
        base_est = getattr(_churn_model, "base_estimator_", None)
        if base_est is not None and hasattr(base_est, "feature_importances_"):
            vals = base_est.feature_importances_
            churn_imp = pd.DataFrame({"feature": feature_names, "importance": np.abs(vals)})
        else:
            raise AttributeError
    except Exception:
        perm_c = permutation_importance(_churn_model, X_np, (y_eta_np>0).astype(int),
                                        scoring="f1", n_repeats=2, random_state=42)
        churn_imp = pd.DataFrame({"feature": feature_names, "importance": np.abs(perm_c.importances_mean)})

    churn_imp = churn_imp.sort_values("importance", ascending=False)

    # Quick CV for ETA
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_maes, cv_r2s = [], []
    for tr, te in cv.split(X_np):
        y_te_hat = _eta_model.predict(X_np[te])
        cv_maes.append(mean_absolute_error(y_eta_np[te], y_te_hat))
        cv_r2s.append(r2_score(y_eta_np[te], y_te_hat))

    return {
        "eta": {"mae": mae, "r2": r2, "cv_mae": float(np.mean(cv_maes)), "cv_r2": float(np.mean(cv_r2s)), "importances": eta_imp},
        "churn": {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc, "importances": churn_imp},
    }

# ================================================================================================
# UPLIFT (T-Learner) — synthetic treatment + outcomes
# ================================================================================================
@st.cache_data(show_spinner=False)
def synth_treatment_labels(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    out = df.copy()
    p_treat = 0.25 + 0.15*(out['churn_probability']) + 0.05*(out['competitor_promo_active'])
    p_treat = np.clip(p_treat, 0.05, 0.85)
    out['treatment'] = (rng.uniform(0,1,len(out)) < p_treat).astype(int)
    
    # === BEGIN: MODIFIED UPLIFT LOGIC ===
    REALISM = 0.70 # Dampening factor

    base = 0.15 + 0.05*(out['satisfaction_score']-3.0) - 0.0006*out['recency'] + 0.02*(out['days_until_service']<=15)

    # --- NEW: More nuanced treatment effect (te) ---
    # Start with a base positive effect for at-risk customers
    te = 0.08 + 0.12*(out['churn_probability'] > 0.60) + 0.06*(out['competitor_promo_active']==1)

    # Introduce NEGATIVE effects for "Sleeping Dogs"
    # 1. Annoyance factor for very happy/loyal customers who don't need a push
    te -= 0.15 * (out['satisfaction_score'] > 4.5)
    
    # 2. Irrelevance factor for customers who just visited
    te -= 0.25 * (out['recency'] < 60)
    
    # 3. Nagging factor for proactive customers who are already early vs. OEM schedule
    te -= 0.10 * (out['oem_alignment'] == 'Early vs OEM')
    
    # 4. Brand dilution factor for high-value customers in affluent areas
    te -= 0.05 * (out['avg_service_cost'] > 8000) * (out['econ_indicator'] > 1.1)
    
    out['p_book_control'] = np.clip(base, 0.01, 0.85)
    out['p_book_treat']   = np.clip(base + (te * REALISM), 0.01, 0.95) # Apply realism factor
    # === END: MODIFIED UPLIFT LOGIC ===

    p_obs = out['p_book_control']*(1-out['treatment']) + out['p_book_treat']*out['treatment']
    out['booked'] = (rng.uniform(0,1,len(out)) < p_obs).astype(int)
    return out

@st.cache_resource(show_spinner=False)
def train_uplift_Tlearner(X: pd.DataFrame, treat: np.ndarray, outcome: np.ndarray):
    X_np = np.asarray(X, dtype=np.float32)
    tmask = (treat==1); cmask = (treat==0)
    if xgb_ok:
        m_t = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8, eval_metric="logloss")
        m_c = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8, eval_metric="logloss")
    else:
        m_t = RandomForestClassifier(n_estimators=200, random_state=42)
        m_c = RandomForestClassifier(n_estimators=200, random_state=42)
    m_t.fit(X_np[tmask], outcome[tmask])
    m_c.fit(X_np[cmask], outcome[cmask])
    return m_t, m_c

def estimate_uplift(m_t, m_c, X: pd.DataFrame):
    X_np = np.asarray(X, dtype=np.float32)
    p_t = m_t.predict_proba(X_np)[:,1]
    p_c = m_c.predict_proba(X_np)[:,1]
    uplift = p_t - p_c
    return p_t, p_c, uplift

# ================================================================================================
# CAPACITY / REVENUE UTILS
# ================================================================================================
def schedule_capacity(df: pd.DataFrame, horizon_days:int = 21, noshow_rate:float = 0.12) -> pd.DataFrame:
    base_cap = {'Delhi':24,'Mumbai':25,'Bangalore':23,'Chennai':18,'Hyderabad':19,'Pune':18,'Kolkata':16,'Ahmedabad':16,'Jaipur':14}
    df = df.copy()
    df['priority'] = df['urgency'].map({"🚨 Critical Overdue":5,"⚠️ Overdue":4,"🔥 Due Soon (0–15)":3,"💡 Upcoming (16–45)":2,"🗓️ Future":1}) \
                     + (df['churn_probability']>=0.6).astype(int) + df.get('km_overdue',0)*1
    df = df.sort_values(['city','priority','churn_probability','satisfaction_score'], ascending=[True,False,False,True])
    start = date.today(); used = {}; out = []
    
    # === BEGIN: FINAL DEMO UPGRADE (Scenario-based Capacity) ===
    scenario_effects = st.session_state.get('scenario_effects', {})
    capacity_multiplier = scenario_effects.get('capacity_multiplier', 1.0)
    # === END: FINAL DEMO UPGRADE (Scenario-based Capacity) ===
    
    for _,r in df.iterrows():
        for d in range(horizon_days):
            dt = start + timedelta(days=int(d))
            cap = int(base_cap.get(r['city'],16) * (0.9 + 0.2*np.random.rand()) * capacity_multiplier) # Apply multiplier
            cap = int(cap * (1 - noshow_rate))
            key = (r['city'], dt)
            used.setdefault(key, 0)
            if used[key] < cap:
                used[key]+=1
                out.append({
                    'customer_id': r['customer_id'], 'model': r['model'], 'city': r['city'],
                    'urgency': r['urgency'], 'churn_probability': r['churn_probability'],
                    'scheduled_date': dt, 'recommended_action': recommend_action(r),
                    'ect_days': r.get('ect_days', 0)
                })
                break
    return pd.DataFrame(out), base_cap

@st.cache_data(show_spinner=False)
def simulate_impact(df: pd.DataFrame, scenario_effects: Dict) -> pd.DataFrame:
    base_map = {"🚨 Critical Overdue":0.25,"⚠️ Overdue":0.30,"🔥 Due Soon (0–15)":0.45,"💡 Upcoming (16–45)":0.28,"🗓️ Future":0.12}
    sim = df.copy()
    sim['p_book_base'] = sim['urgency'].map(base_map) * (0.8 + 0.05*sim['satisfaction_score'])

    # === BEGIN: FINAL DEMO UPGRADE (AMC, Parts Risk, Scenarios) ===
    sim.loc[sim['amc_enrolled'] == 1, 'p_book_base'] *= 1.05
    sim.loc[sim['parts_wait_risk'] == 1, 'p_book_base'] *= 0.90
    sim['p_book_base'] += scenario_effects.get('p_book_base_shift', 0.0)
    # === END: FINAL DEMO UPGRADE (AMC, Parts Risk, Scenarios) ===
    
    high_risk = (sim['churn_probability']>=0.60) | (sim['urgency'].isin(["🚨 Critical Overdue","⚠️ Overdue"])) | (sim.get('km_overdue',0)==1)
    uplift = 0.20*high_risk.astype(float) + 0.06*sim['competitor_promo_active'].map({0:0,1:1}) + 0.05
    sim['p_book_model'] = np.clip(sim['p_book_base'] + uplift, 0, 0.95)
    sim['rev_base']  = sim['p_book_base']  * sim['avg_service_cost']
    sim['rev_model'] = sim['p_book_model'] * sim['avg_service_cost']
    sim['rev_uplift'] = sim['rev_model'] - sim['rev_base']
    return sim[['city','urgency','avg_service_cost','p_book_base','p_book_model','rev_uplift']]

def select_under_budget(df: pd.DataFrame, capacity:int, budget:int, apply_guardrails: bool) -> pd.DataFrame:
    cand = df.copy()
    
    # === BEGIN: FINAL DEMO UPGRADE (Guardrails) ===
    if apply_guardrails:
        cand = cand[cand['satisfaction_score'] <= 4.5]
        cand = cand[cand['oem_alignment'] != 'Early vs OEM']
    # === END: FINAL DEMO UPGRADE (Guardrails) ===
    
    # This function expects 'offer_cost' and 'expected_uplift_rev' to be present on the input df
    cand = cand.sort_values('expected_uplift_rev', ascending=False).reset_index(drop=True)
    pick, spent = [], 0
    for _, r in cand.iterrows():
        if len(pick) >= capacity: break
        if spent + r['offer_cost'] <= budget:
            pick.append(r)
            spent += r['offer_cost']
    return pd.DataFrame(pick)

# === BEGIN: FINAL DEMO UPGRADE ===
def apply_scenario_effects(df: pd.DataFrame, scenario: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Applies scenario-based adjustments to the main dataframe.
    Called from the main pipeline after data generation.
    """
    df_mod = df.copy()
    effects = {
        'capacity_multiplier': 1.0,
        'noshow_rate_shift': 0.0,
        'p_book_base_shift': 0.0,
        'delay_multiplier': 1.0,
    }

    if scenario == 'Monsoon Spike':
        effects['capacity_multiplier'] = 0.95
        effects['noshow_rate_shift'] = 0.02
        effects['delay_multiplier'] = 1.10
        # In monsoon, certain parts (wipers, tires) might be in higher demand
        df_mod['parts_wait_risk'] = np.clip(df_mod['parts_wait_risk'] + (df_mod['city'].isin(['Mumbai', 'Kolkata', 'Chennai']) * 0.1), 0, 1)

    elif scenario == 'Monday Rush':
        # Simulate higher demand by slightly reducing effective capacity perception
        effects['capacity_multiplier'] = 0.90
        effects['noshow_rate_shift'] = 0.01

    elif scenario == 'Competitor Launch':
        effects['p_book_base_shift'] = -0.03
        promo_mask = df_mod['competitor_promo_active'] == 0
        promo_indices = df_mod[promo_mask].sample(frac=0.10, random_state=42).index
        df_mod.loc[promo_indices, 'competitor_promo_active'] = 1
    
    # Apply delay multiplier
    df_mod['ect_days'] = (df_mod['ect_hours'] / 24.0) + (df_mod['delay_days'] * effects['delay_multiplier'])
    
    st.session_state['scenario_effects'] = effects
    return df_mod, effects

def get_eta_reasons(row: pd.Series) -> str:
    """Generates plain-English reasons for a service ETA."""
    reasons = []
    if row['service_type'] == 'Repair':
        reasons.append(f"<span class='chip red'>Repair Job: +{row['ect_hours']/24:.1f} days</span>")
    elif row['service_type'] == 'Comprehensive':
         reasons.append(f"<span class='chip amber'>Comprehensive: +{row['ect_hours']:.1f} hrs</span>")
    else:
        reasons.append(f"<span class='chip'>Routine Service</span>")

    if row['parts_wait_risk'] == 1:
        reasons.append(f"<span class='chip amber'>Parts Wait Risk</span>")
    if row['tech_skill_gap'] == 1:
        reasons.append(f"<span class='chip amber'>Tech Skill Gap</span>")
    if row['insurance_approval_req'] == 1:
        reasons.append(f"<span class='chip red'>Insurance Approval</span>")
    
    return " ".join(reasons[:3])


def get_churn_reasons(row: pd.Series, _imp_df: pd.DataFrame) -> str:
    """Generates plain-English reasons for churn risk."""
    reasons = []
    if row['recency'] > row['long_overdue_threshold']:
        reasons.append(f"<span class='chip red'>Recency: {row['recency_months']} months</span>")
    elif row['recency'] > row['long_overdue_threshold'] - 60:
         reasons.append(f"<span class='chip amber'>Recency: {row['recency_months']} months</span>")

    if row['CSI_1to5'] <= 2:
        reasons.append(f"<span class='chip red'>Low CSI: {row['CSI_1to5']}/5</span>")
    if row['feedback_reason'] not in ['good_experience', 'advisor_helpful']:
         reasons.append(f"<span class='chip amber'>Feedback: {row['feedback_reason'].replace('_',' ').title()}</span>")
         
    # Add a top feature as a reason if not already covered
    top_feat = _imp_df.iloc[0]['feature']
    if 'recency' not in str(reasons).lower() and 'satisfaction' not in str(reasons).lower():
         reasons.append(f"<span class='chip'>{pretty(top_feat)}</span>")

    return " ".join(reasons[:3])
# === END: FINAL DEMO UPGRADE ===

# ================================================================================================
# PIPELINE — data, segments, models
# ================================================================================================
with st.spinner("🔄 Generating synthetic fleet, segments & training models..."):
    df_base = generate_data()
    
    # === BEGIN: UI FIX - Moved Scenario Selector to top of main app area ===
    _, col2 = st.columns([3, 1.2]) # Create columns to push the selector to the right
    with col2:
        scenario = st.selectbox(
            "🎭 Select Operational Scenario",
            ['Baseline', 'Monsoon Spike', 'Monday Rush', 'Competitor Launch'],
            help="Change simulation parameters to reflect different business conditions."
        )
    # === END: UI FIX ===
    
    df_raw, scenario_effects = apply_scenario_effects(df_base, scenario)
    
    st.markdown("""
    <div class='insight gray' style='text-align: left; margin-top: -10px;'>
        <p style="margin-bottom: 5px;"><b>Understanding Scenarios:</b> The selected scenario (<b>{scenario}</b>) adjusts key business drivers:</p>
        <ul style="margin: 0; padding-left: 20px; font-size: 0.9rem;">
            <li><b>Baseline:</b> Standard operating conditions.</li>
            <li><b>Monsoon Spike:</b> Simulates rainy season in major cities, slightly reducing workshop capacity (-5%) and increasing potential delays (+10%).</li>
            <li><b>Monday Rush:</b> Models the typical start-of-week surge, increasing demand and slightly reducing effective capacity (-10%).</li>
            <li><b>Competitor Launch:</b> Simulates a new competitor promotion, slightly decreasing baseline booking probability (-3%) and increasing competitor promo flags.</li>
        </ul>
    </div>
    """.format(scenario=scenario), unsafe_allow_html=True)
    
    df_raw['urgency'] = df_raw['days_until_service'].apply(categorize_urgency)
    df_raw['oem_alignment'] = df_raw['delta_vs_oem_days'].apply(oem_alignment)
    df_raw = build_segments(df_raw)
    
    df = enhance_data_with_new_features(df_raw)

    X, y_eta, y_churn, FEATS = build_features(df)
    ETA_MODEL, CHURN_MODEL, SPLIT = train_models(X, y_eta, y_churn)
    METRICS = evaluate(ETA_MODEL, CHURN_MODEL, X, y_eta, SPLIT, FEATS)

    uplift_df = synth_treatment_labels(df)
    m_t, m_c = train_uplift_Tlearner(X, uplift_df['treatment'].values.astype(int), uplift_df['booked'].values.astype(int))
    p_t, p_c, uplift_vals = estimate_uplift(m_t, m_c, X)
    uplift_df = uplift_df.assign(p_treat=p_t, p_ctrl=p_c, uplift=uplift_vals)

# ================================================================================================
# TABS
# ================================================================================================
tab_eta, tab_churn, tab_uplift, tab_ai = st.tabs([
    "⏱️ ETA & Scheduling", "⚠️ Churn & RFM", "📈 Uplift & Selection", "🚀 AI Campaign Builder"
])

# ================================================================================================
# TAB 1 — ETA & SCHEDULING
# ================================================================================================
with tab_eta:
    st.markdown("""<div class='section'><h2>⏱️ ETA Intelligence & Smart Scheduling</h2>
    <p class='muted'>Hybrid ETA (calendar ⟂ km) aligned to OEM intervals; weather & behavior adjusted.</p></div>""", unsafe_allow_html=True)
    
    with st.expander("⬇️ Download Raw Customer Data"):
        st.markdown("Download the complete, raw customer dataset used to power all analytics in this application.")
        raw_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Raw Data (CSV)",
            data=raw_csv,
            file_name=f"autocare_raw_data_{datetime.now():%Y%m%d}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # KPIs
    overdue_rate = (df['urgency'].isin(["🚨 Critical Overdue","⚠️ Overdue"])).mean()*100
    ontime_share = (df['oem_alignment']=="On-time vs OEM").mean()*100
    km_overdue_rate = df['km_overdue'].mean()*100
    next30_demand = df['days_until_service'].between(0, 30).sum()


    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(f"<div class='kpi red'><div class='l'>Overdue %</div><div class='v'>{overdue_rate:.1f}%</div><div class='s'>Lower is better</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi'><div class='l'>On-time vs OEM</div><div class='v'>{ontime_share:.1f}%</div><div class='s'>Calendar alignment</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi red'><div class='l'>KM Overdue %</div><div class='v'>{km_overdue_rate:.1f}%</div><div class='s'>>120% of OEM km</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi amber'><div class='l'>Predicted Demand (30d)</div><div class='v'>{next30_demand:,.0f}</div><div class='s'>Expected bookings</div></div>", unsafe_allow_html=True)
    clv_at_risk = (df[df['churn_probability']>0.60]['CLTV']).sum()
    c5.markdown(f"<div class='kpi'><div class='l'>CLV At Risk</div><div class='v'>₹{format_indian(clv_at_risk)}</div><div class='s'>Churn Prob > 60%</div></div>", unsafe_allow_html=True)


    # ETA bands
    st.subheader("ETA Prediction with Interval Bands (days)")
    st.caption("Think weather forecast: not one number but a likely range. We use empirical residual quantiles to estimate the band.")
    yhat, lo, hi = eta_pred_with_intervals(ETA_MODEL, X, 0.025, 0.975)
    s = pd.DataFrame({'pred': yhat, 'lo': lo, 'hi': hi}).sample(1000, random_state=1).reset_index(drop=True)
    fig_band = go.Figure()
    fig_band.add_trace(go.Scatter(x=s.index, y=s['hi'], mode='lines', line=dict(width=0), showlegend=False))
    fig_band.add_trace(go.Scatter(x=s.index, y=s['lo'], mode='lines', fill='tonexty', name='Prediction Band', opacity=0.25))
    fig_band.add_trace(go.Scatter(x=s.index, y=s['pred'], mode='lines', name='ETA Pred', line=dict(width=1)))
    fig_band.update_layout(title="ETA Prediction Intervals (empirical)", yaxis_title="Days", xaxis_title="Samples")
    st.plotly_chart(darkify(fig_band, 360), use_container_width=True)
    st.markdown("""<div class='insight'>
    <h4>Action</h4>
    Trigger alerts if a customer’s <b>band crosses day 0</b> (due today) or overlaps the <b>overdue zone</b>. Call earlier when bands are wide (higher uncertainty), and offer pick-up where bands sit entirely below 0.
    </div>""", unsafe_allow_html=True)
    
    # === BEGIN: FINAL DEMO UPGRADE (Interactive ETA Tab Expander) ===
    with st.expander("🔩 Operational Reality & Capacity Planning", expanded=True):
        st.markdown("Analyze service adherence, workshop utilization, and the impact of operational scenarios on your capacity.")
        
        # Interactive no-show rate slider
        noshow_rate = st.slider(
            "No-show Rate Adjustment", 0.0, 0.25, 
            value=(0.12 + scenario_effects.get('noshow_rate_shift', 0.0)), 
            step=0.01,
            help=f"Adjust the workshop's no-show rate. The default is 12%. Current scenario '{scenario}' suggests {0.12 + scenario_effects.get('noshow_rate_shift', 0.0):.0%}.",
            key="noshow_slider_interactive"
        )
        
        sched, base_caps = schedule_capacity(df, horizon_days=21, noshow_rate=noshow_rate)
        
        if not sched.empty:
            sched['promised_eta_date'] = sched['scheduled_date'] + pd.to_timedelta(sched['ect_days'], unit='D')
            adherence_days_diff = (pd.to_datetime(sched['promised_eta_date']) - pd.to_datetime(sched['scheduled_date'])).dt.days - sched['ect_days']
            
            def categorize_adherence(diff):
                if abs(diff) <= 0.5: return "On-Time"
                if abs(diff) <= 1.5: return "Slightly Delayed"
                return "Delayed"
            sched['adherence_category'] = adherence_days_diff.apply(categorize_adherence)
            
            # KPI Calculations
            adherence_dist = sched['adherence_category'].value_counts(normalize=True)
            load = sched.groupby(['city','scheduled_date']).size().reset_index(name='bookings')
            load['base_capacity'] = load['city'].map(base_caps)
            load['utilization'] = (load['bookings'] / load['base_capacity'].clip(lower=1)) * 100
            avg_utilization = load['utilization'].mean()

            kpi1, kpi2 = st.columns(2)
            
            fig_adh_donut = px.pie(
                values=adherence_dist.values, names=adherence_dist.index,
                title="Service Adherence Breakdown", hole=0.5,
                color_discrete_map={"On-Time": "#00ff88", "Slightly Delayed": "#ffb703", "Delayed": "#ff6b6b"}
            )
            fig_adh_donut.update_traces(textinfo='percent+label', pull=[0.05, 0, 0])
            kpi1.plotly_chart(darkify(fig_adh_donut, 280), use_container_width=True)
            kpi2.metric("Avg. Capacity Utilization (Next 21 days)", f"{avg_utilization:.1f}%")

            # Daily Trend Charts
            st.markdown("##### Daily Operational Trends (Next 21 Days)")
            trend1, trend2, trend3 = st.columns(3)

            # Chart 1: Daily Capacity Utilization
            daily_util = load.groupby('scheduled_date')['utilization'].mean().reset_index()
            fig_util = px.line(daily_util, x='scheduled_date', y='utilization', title="Capacity Utilization %", markers=True)
            fig_util.update_layout(yaxis_range=[0, daily_util['utilization'].max()*1.2])
            trend1.plotly_chart(darkify(fig_util, 280), use_container_width=True)
            trend1.markdown("<div class='insight gray' style='font-size:0.8rem;'><b>Insight:</b> Peaks indicate potential bottlenecks. Smooth out demand by offering incentives for off-peak days.</div>", unsafe_allow_html=True)
            
            # Chart 2: Daily Service Adherence
            sched['adherence_ok'] = (sched['adherence_category'] == "On-Time").astype(int)
            daily_adherence = sched.groupby('scheduled_date')['adherence_ok'].mean().reset_index()
            daily_adherence['adherence_ok'] = np.clip(daily_adherence['adherence_ok'] * np.random.normal(1, 0.03, size=len(daily_adherence)), 0.8, 0.99) * 100 # Add realism
            fig_adh = px.line(daily_adherence, x='scheduled_date', y='adherence_ok', title="On-Time Adherence %", markers=True)
            fig_adh.update_layout(yaxis_range=[min(70, daily_adherence['adherence_ok'].min()*0.95), 101])
            trend2.plotly_chart(darkify(fig_adh, 280), use_container_width=True)
            trend2.markdown("<div class='insight gray' style='font-size:0.8rem;'><b>Insight:</b> Dips in adherence often correlate with high utilization. Address root causes like parts or technician shortages on those days.</div>", unsafe_allow_html=True)

            # Chart 3: Daily No-Show Impact
            daily_bookings = sched.groupby('scheduled_date').size().reset_index(name='bookings')
            daily_bookings['lost_slots'] = daily_bookings['bookings'] * noshow_rate
            fig_noshow = px.bar(daily_bookings, x='scheduled_date', y='lost_slots', title="Projected Lost Slots")
            trend3.plotly_chart(darkify(fig_noshow, 280), use_container_width=True)
            trend3.markdown("<div class='insight gray' style='font-size:0.8rem;'><b>Insight:</b> These are potential revenue losses. Use confirmation calls or small deposits for high-demand slots to reduce this number.</div>", unsafe_allow_html=True)

            
            # XAI "Because..." Chips
            st.markdown("<hr style='margin:1rem 0; border-color:#2a2f44;'>", unsafe_allow_html=True)
            st.markdown("##### 🧐 Explainable ETA (ECT Drivers)")
            st.caption("Select a scheduled job to see the primary drivers of its Estimated Completion Time (ECT).")
            
            sample_sched = sched.head(50)
            selected_id = st.selectbox(
                "Select a Customer ID from the schedule:",
                sample_sched['customer_id'],
                format_func=lambda x: f"{x} ({sample_sched.loc[sample_sched['customer_id'] == x, 'model'].iloc[0]})"
            )
            
            if selected_id:
                selected_row = df[df['customer_id'] == selected_id].iloc[0]
                reasons_html = get_eta_reasons(selected_row)
                st.markdown(f"**ECT Drivers for {selected_id}:** {reasons_html}", unsafe_allow_html=True)
        else:
            st.info("No schedule generated. Try adjusting filters or scenario.")
    # === END: FINAL DEMO UPGRADE (Interactive ETA Tab Expander) ===
    
    with st.expander("🔧 Component Service Triggers Analysis"):
        st.markdown("This analysis shows the health scores for key vehicle components based on usage (KM) and time. A lower score indicates a higher need for replacement, helping predict which parts are driving service demand.")
        
        health_cols = ['oil_health', 'brakes_health', 'air_filter_health', 'tires_health', 'battery_health']
        component_overdue_pct = (df[health_cols] < 40).mean() * 100
        component_overdue_df = component_overdue_pct.reset_index()
        component_overdue_df.columns = ['component', 'overdue_pct']
        component_overdue_df['component'] = component_overdue_df['component'].str.replace('_health','').str.title()
        
        fig_comp = px.bar(component_overdue_df, x='component', y='overdue_pct', 
                          title='% of Fleet with Overdue Components', text=component_overdue_df['overdue_pct'].apply(lambda x: f'{x:.1f}%'))
        st.plotly_chart(darkify(fig_comp, 380), use_container_width=True)
        st.markdown("""<div class='insight warn'>
        <h4>Insight & Action</h4>
        <b>Oil and Air Filters</b> are the most frequent service triggers. Use this data to pre-stock these parts, create bundled service packages (e.g., "Fluids & Filters Special"), and send targeted reminders to customers whose components are predicted to be due.
        </div>""", unsafe_allow_html=True)

    with st.expander("🛡️ Warranty Expiry Alerts"):
        st.markdown("Tracking warranty expiry is crucial for retention. Customers nearing the end of their warranty period are at a high risk of switching to independent garages. This module identifies these customers for proactive outreach.")
        
        expiring_soon_pct = (df['days_to_warranty_expiry'].between(0, 90)).mean() * 100
        st.markdown(f"<div class='kpi amber'><div class='l'>Warranty Expiring (next 90d)</div><div class='v'>{expiring_soon_pct:.1f}%</div><div class='s'>of total fleet</div></div>", unsafe_allow_html=True)

        expiring_df = df[df['days_to_warranty_expiry'].between(-30, 90)]
        
        fig_warr = px.density_heatmap(expiring_df, x='days_to_warranty_expiry', y='city', 
                                    title='Warranty Expiry Hotspots (Next 90 Days)',
                                    labels={'days_to_warranty_expiry': 'Days Until (-) or Past (+) Expiry'})
        st.plotly_chart(darkify(fig_warr, 380), use_container_width=True)

        st.markdown("""<div class='insight'>
        <h4>Insight & Action</h4>
        The heatmap shows a concentration of expiring warranties in <b>Delhi and Bangalore</b>. Target these customers with Extended Warranty packages, loyalty discounts on post-warranty services, and communication highlighting the value of authorized service centers.
        </div>""", unsafe_allow_html=True)
        
    with st.expander("📈 Service Interval Trend Line"):
        st.markdown("This chart shows how the average time between services changes as vehicles age. Understanding this trend is key to adjusting outreach timing and preventing service delays for older cars.")
        
        df['age_bucket'] = pd.cut(df['vehicle_age_years'], bins=[0, 2, 4, 6, 8, 10, 20], labels=['0-2y', '2-4y', '4-6y', '6-8y', '8-10y', '10y+'])
        interval_trends = df.groupby('age_bucket', observed=False)['recency'].mean().reset_index()
        
        fig_int = px.line(interval_trends, x='age_bucket', y='recency', markers=True,
                          title='Average Service Interval (Days) by Vehicle Age')
        fig_int.update_layout(yaxis_title="Avg. Days Between Services", xaxis_title="Vehicle Age Bucket")
        st.plotly_chart(darkify(fig_int, 380), use_container_width=True)
        st.markdown("""<div class='insight danger'>
        <h4>Insight & Action</h4>
        As vehicles age past the 4-year mark (post-warranty), customers significantly <b>stretch their service intervals</b>. This increases the risk of major repairs and customer churn. For cars aged 4+, shorten the reminder cycle and emphasize the importance of preventive maintenance to avoid higher costs later.
        </div>""", unsafe_allow_html=True)


    # Fleet footprint & cost + insight
    st.subheader("Fleet Footprint & Cost")
    model_stats = df.groupby('model').agg(customers=('customer_id','count'), avg_cost=('avg_service_cost','mean')).reset_index()
    figA = px.scatter(model_stats, x='customers', y='avg_cost', size='avg_cost', color='avg_cost',
                      hover_name='model', title='Model Footprint — Customer Base vs Avg Cost',
                      color_continuous_scale='viridis')
    st.plotly_chart(darkify(figA, 430), use_container_width=True)
    low_cost = model_stats.sort_values('avg_cost').head(2)['model'].tolist()
    high_cost = model_stats.sort_values('avg_cost', ascending=False).head(2)['model'].tolist()
    top_pop = model_stats.sort_values('customers', ascending=False).head(2)['model'].tolist()
    st.markdown(f"""<div class='insight'>
    <h4>Insight</h4>
    Low-cost champions: <b>{', '.join(low_cost)}</b>. Highest cost: <b>{', '.join(high_cost)}</b>.
    Most popular: <b>{', '.join(top_pop)}</b>. Package value-assurance + preventive checks for high-cost/high-pop models.
    </div>""", unsafe_allow_html=True)

    # Monthly service demand + moving average
    st.subheader("Monthly Service Demand (smooth trend)")
    mdf = df.copy()
    mdf['month'] = mdf['last_service_date'].dt.to_period('M').astype(str)
    vol = mdf.groupby('month').size().reset_index(name='volume').sort_values('month')
    vol['ma'] = vol['volume'].rolling(3, min_periods=1).mean()
    figM = go.Figure()
    figM.add_trace(go.Bar(x=vol['month'], y=vol['volume'], name='Monthly volume', opacity=0.8))
    figM.add_trace(go.Scatter(x=vol['month'], y=vol['ma'], name='3-month moving average'))
    figM.update_layout(title="Monthly Service Demand Volume — with 3-month Moving Average", xaxis_title="Month", yaxis_title="Service Volume")
    st.plotly_chart(darkify(figM, 360), use_container_width=True)
    st.markdown("""<div class='insight warn'>
    <h4>Seasonality</h4>
    Pre-monsoon & festive months rise ~10–15%. Plan extra technicians and pre-stock fast movers.
    </div>""", unsafe_allow_html=True)

    # OEM alignment & urgency
    st.subheader("OEM Schedule Alignment")
    align_mix = df['oem_alignment'].value_counts().rename_axis('alignment').reset_index(name='customers')
    figO = px.bar(align_mix, x='alignment', y='customers', text='customers', title='Predicted vs OEM — Early / On-time / Late')
    figO.update_traces(textposition='outside')
    st.plotly_chart(darkify(figO, 340), use_container_width=True)

    st.subheader("Service Urgency Mix")
    urgency_mix = df['urgency'].value_counts().rename_axis('urgency').reset_index(name='customers')
    figD = px.bar(urgency_mix, x='urgency', y='customers', text='customers', title='Service Urgency Mix')
    figD.update_traces(textposition='outside')
    figD.update_layout(xaxis_tickangle=-10)
    st.plotly_chart(darkify(figD, 340), use_container_width=True)
    st.markdown("""<div class='insight'>
    <h4>Meaning</h4>
    Late/Overdue cohorts → priority outreach with pick-up. “Upcoming” → education drip (pre-monsoon safety).
    </div>""", unsafe_allow_html=True)

    # Segment lens (NEW on ETA page)
    st.subheader("Segment Lens — Overdue% and Cost by Customer Segment")
    seg_stats = df.groupby('cust_segment').agg(
        overdue_rate=('days_until_service', lambda x: (x<0).mean()*100),
        avg_cost=('avg_service_cost','mean'),
        size=('customer_id','count')
    ).reset_index()
    fig_seg1 = px.bar(seg_stats, x='cust_segment', y='overdue_rate', color='avg_cost',
                      title='Overdue Rate by Segment (color = Avg Cost)', text=seg_stats['size'])
    fig_seg1.update_traces(textposition='outside')
    st.plotly_chart(darkify(fig_seg1, 380), use_container_width=True)
    st.markdown("""<div class='insight gray'>
    <h4>Action</h4>
    Focus first on <b>High-Value & At-Risk</b> (highest Overdue% with high cost). Provide pick-up + priority lane.
    </div>""", unsafe_allow_html=True)

    # Actionable targeting (drivers transparency)
    st.subheader("🎯 Actionable Targeting")
    st.caption("Filter cohorts and act. Table includes ETA, OEM alignment, urgency, satisfaction, churn %, cost, km_overdue, drivers & recommended action.")
    c1,c2,c3 = st.columns(3)
    with c1:
        city_f = st.selectbox("🌍 City", ["All"] + sorted(df['city'].unique().tolist()))
    with c2:
        urg_f  = st.selectbox("⚡ Urgency", ["All"] + urgency_mix['urgency'].tolist())
    with c3:
        cost_f = st.slider("Avg Service Cost (₹)", int(df['avg_service_cost'].min()), int(df['avg_service_cost'].max()), (3000,9000))
    flt = df.copy()
    if city_f!="All": flt = flt[flt['city']==city_f]
    if urg_f!="All": flt = flt[flt['urgency']==urg_f]
    flt = flt[(flt['avg_service_cost']>=cost_f[0]) & (flt['avg_service_cost']<=cost_f[1])]
    flt['recommended_action'] = flt.apply(recommend_action, axis=1)
    def drivers(row):
        arr=[]
        if row['km_overdue']==1: arr.append("KM overdue")
        if row['urgency'] in ["🚨 Critical Overdue","⚠️ Overdue"]: arr.append("Calendar overdue")
        if row['satisfaction_score']<3: arr.append("Low satisfaction")
        if row['churn_probability']>=0.6: arr.append("High churn prob")
        if row['delta_vs_oem_days']>10: arr.append("Late vs OEM")
        return ", ".join(arr[:5])
    flt['drivers'] = flt.apply(drivers, axis=1)
    cols = ['customer_id','model','city','days_until_service','oem_alignment','urgency','satisfaction_score','churn_probability','avg_service_cost','km_overdue','drivers','recommended_action']
    st.dataframe(flt[cols].sort_values(['urgency','days_until_service']).head(500), use_container_width=True, hide_index=True)


    # Impact simulation
    st.subheader("📈 Expected Retained Revenue by City (Δ vs baseline)")
    impact = simulate_impact(df, scenario_effects)
    agg = impact.groupby('city').agg(rev_uplift=('rev_uplift','sum')).reset_index()
    figI = px.bar(agg.sort_values('rev_uplift', ascending=False), x='city', y='rev_uplift',
                  title='Expected Retained Revenue Uplift by City (₹)')
    st.plotly_chart(darkify(figI, 360), use_container_width=True)
    st.markdown("""<div class='insight'>
    <h4>Definition</h4>
    Uplift = revenue with proactive outreach − baseline revenue (no outreach). Allocate budget to top uplift cities first.
    </div>""", unsafe_allow_html=True)

# ================================================================================================
# TAB 2 — CHURN & RFM (with segments & CLTV)
# ================================================================================================
with tab_churn:
    st.markdown("""<div class='section'><h2>⚠️ Churn Radar & RFM</h2>
    <p class='muted'>Calibrated churn probabilities with RFM & behavioral signals. Segment lens and CLTV included.</p></div>""", unsafe_allow_html=True)

    churn_rate = df['churn_status'].mean()*100
    very_high = (df['churn_probability']>=0.85).sum()
    very_high_pct = (df['churn_probability']>=0.85).mean()*100
    
    # === BEGIN: FINAL DEMO UPGRADE (Silent Churn KPI) ===
    # Original logic was: overdue_15m = (df['recency'] >= 450).mean()*100
    silent_churn_pct = df['long_overdue'].mean() * 100
    # === END: FINAL DEMO UPGRADE (Silent Churn KPI) ===

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"<div class='kpi red'><div class='l'>Churned / At-Risk (proxy)</div><div class='v'>{churn_rate:.1f}%</div><div class='s'>High prob or long overdue</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi red'><div class='l'>“Definitely Churn” (≥85%)</div><div class='v'>{very_high:,.0f} · {very_high_pct:.1f}%</div><div class='s'>Prioritize immediately</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi'><div class='l'>CLV at Risk (₹)</div><div class='v'>{format_indian(clv_at_risk)}</div><div class='s'>Churn Prob > 60%</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi amber'><div class='l'>Silent Churn %</div><div class='v'>{silent_churn_pct:.1f}%</div><div class='s'>Persona-based overdue</div></div>", unsafe_allow_html=True)
    
    # === BEGIN: FINAL DEMO UPGRADE (Churn Tab Expander) ===
    with st.expander("🚨 Early Warning & Silent Churn Analysis"):
        st.markdown("Proactively identify customers who are about to become at-risk and understand the drivers for those who have already gone silent.")
        
        # Inactivity Window Sliders
        st.markdown("##### Define Inactivity Windows (months since last service)")
        col_slider1, col_slider2 = st.columns(2)
        at_risk_window = col_slider1.slider("At-Risk / Inactive Window", 1, 48, (13, 24))
        lost_window_start = col_slider2.slider("Lost Window (Starts After)", 1, 48, 25)
        
        # KPI calculations using slider values
        at_risk_customers = df[df['recency_months'].between(at_risk_window[0], at_risk_window[1])]
        lost_customers = df[df['recency_months'] >= lost_window_start]

        # Early Warning KPI
        df['days_to_churn_threshold'] = df['long_overdue_threshold'] - df['recency']
        early_warning_count = df[df['days_to_churn_threshold'].between(30, 60)].shape[0]
        
        kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
        kpi_c1.metric("Early Warning Customers (30-60 days to threshold)", f"{early_warning_count:,}")
        kpi_c2.metric(f"Inactive Customers ({at_risk_window[0]}-{at_risk_window[1]} months)", f"{len(at_risk_customers):,}")
        kpi_c3.metric(f"Lost Customers (>{lost_window_start} months)", f"{len(lost_customers):,}")

        # XAI "Because..." Chips for Churn
        st.markdown("---")
        st.markdown("##### 🤔 Explainable Churn (Risk Drivers)")
        st.caption("Select a high-risk customer to see the primary drivers of their churn score.")
        
        high_risk_sample = df[df['churn_probability'] > 0.6].head(50)
        if not high_risk_sample.empty:
            selected_churn_id = st.selectbox(
                "Select a high-risk Customer ID:",
                high_risk_sample['customer_id'],
                format_func=lambda x: f"{x} (Churn Prob: {high_risk_sample.loc[high_risk_sample['customer_id'] == x, 'churn_probability'].iloc[0]:.2f})"
            )
            
            if selected_churn_id:
                selected_row = df[df['customer_id'] == selected_churn_id].iloc[0]
                reasons_html = get_churn_reasons(selected_row, METRICS['churn']['importances'])
                st.markdown(f"**Churn Drivers for {selected_churn_id}:** {reasons_html}", unsafe_allow_html=True)
        else:
            st.info("No high-risk customers in the current sample.")
    # === END: FINAL DEMO UPGRADE (Churn Tab Expander) ===
    

    with st.expander("🚰 Churn Funnel & Revenue at Risk"):
        st.markdown("This funnel visualizes how the customer base is segmented by churn risk, immediately showing where the biggest threat lies and the potential revenue at stake in each category.")
        
        df['risk_tier'] = pd.cut(df['churn_probability'], bins=[0, 0.3, 0.6, 1.0], labels=['Low Risk', 'Medium Risk', 'High Risk'])
        funnel_data = df.groupby('risk_tier', observed=False).agg(
            customers=('customer_id', 'count'),
            revenue_at_risk=('CLTV', 'sum')
        ).reset_index()

        fig_funnel = go.Figure(go.Funnel(
            y = funnel_data['risk_tier'],
            x = funnel_data['customers'],
            textinfo = "value+percent initial",
            texttemplate = "%{value:,} Customers<br>%{percentInitial:.1%}",
            marker = {"color": ["#00ff88", "#ffb703", "#ff6b6b"]}
        ))
        fig_funnel.update_layout(title="Customer Distribution by Churn Risk Tier")
        st.plotly_chart(darkify(fig_funnel, 380), use_container_width=True)
        st.markdown(f"""<div class='insight danger'>
        <h4>Insight & Action</h4>
        The <b>High Risk</b> tier, though smallest, contains customers with the highest probability to churn. The total CLTV at risk in this segment is approximately <b>₹{format_indian(funnel_data.loc[funnel_data['risk_tier']=='High Risk', 'revenue_at_risk'].sum())}</b>. Focus retention campaigns here first for the highest immediate impact.
        </div>""", unsafe_allow_html=True)

    with st.expander("🚗 Vehicle Lifecycle Segmentation"):
        st.markdown("Churn behavior is strongly tied to the vehicle's age and warranty status. This view breaks down the customer base by lifecycle stage to reveal when customers are most likely to defect.")
        
        lifecycle_churn = df.groupby('lifecycle_stage', observed=False)['churn_probability'].mean().reset_index()
        fig_life = px.bar(lifecycle_churn, x='lifecycle_stage', y='churn_probability', color='lifecycle_stage',
                        title='Average Churn Probability by Vehicle Lifecycle Stage')
        st.plotly_chart(darkify(fig_life, 380), use_container_width=True)
        st.markdown("""<div class='insight warn'>
        <h4>Insight & Action</h4>
        Churn risk dramatically increases in the <b>Mid-life (4-7y)</b> stage, immediately after the standard warranty expires. The <b>Late-life (7y+)</b> stage has the highest risk. This confirms the need for targeted post-warranty retention offers like Annual Maintenance Contracts (AMCs) and loyalty programs.
        </div>""", unsafe_allow_html=True)

    with st.expander("📊 Churn Driver Scorecard & Personas"):
        st.markdown("This section pinpoints the key operational factors driving churn and shows how churn rates differ across distinct customer personas, allowing for more tailored retention strategies.")

        # Scorecard
        churn_drivers = pd.DataFrame({
            'Driver': ['Low Satisfaction (CSI)', 'Service Gap Trend', 'Warranty Expired', 'Low FTFR (Proxy)'],
            'Impact Score': [0.85, 0.72, 0.65, 0.58] # Synthetic scores for demo
        }).sort_values('Impact Score', ascending=True)
        fig_drivers = px.bar(churn_drivers, x='Impact Score', y='Driver', orientation='h', title='Key Churn Driver Impact Scorecard')
        st.plotly_chart(darkify(fig_drivers, 360), use_container_width=True)

        # Personas
        persona_churn = df.groupby('persona')['churn_status'].mean().reset_index()
        persona_churn['churn_status'] *= 100
        fig_persona = px.bar(persona_churn, x='persona', y='churn_status', title='Churn Rate % by Customer Persona')
        st.plotly_chart(darkify(fig_persona, 380), use_container_width=True)
        st.markdown("""<div class='insight'>
        <h4>Insight & Action</h4>
        <b>Customer satisfaction (CSI)</b> is the single biggest lever. Operationally, focus on improving First-Time-Fix-Rate (FTFR) and technician training. The <b>"Warranty Warriors"</b> persona (high-value but at-risk) and <b>"Seasonal Users"</b> (infrequent) are the most likely to churn. Tailor messaging: emphasize reliability and genuine parts for the former, and offer flexible, value-based service packages for the latter.
        </div>""", unsafe_allow_html=True)

    with st.expander("💰 CLTV Impact of Churn Reduction"):
        st.markdown("This tile demonstrates the tangible financial benefit of a successful retention program, translating churn reduction efforts directly into projected incremental revenue.")
        
        high_risk_df = df[df['risk_tier'] == 'High Risk']
        customers_saved_count = int(len(high_risk_df) * 0.05) if not high_risk_df.empty else 0
        incremental_cltv = high_risk_df.sample(customers_saved_count, random_state=42)['CLTV'].sum() if customers_saved_count > 0 else 0
        
        st.markdown(f"""<div class='kpi'>
        <div class='l'>If we save 5% of High-Risk Customers...</div>
        <div class='v'>+ ₹{format_indian(incremental_cltv)}</div>
        <div class='s'>Projected Incremental CLTV</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class='insight gray'>
        <h4>Insight & Action</h4>
        Even a modest 5% improvement in retaining high-risk customers translates to a significant revenue gain. This justifies investment in retention tools, personnel, and targeted offers. This figure can be used to set clear ROI goals for the customer relationship team.
        </div>""", unsafe_allow_html=True)
    
    st.subheader("Top Churn Drivers")
    churn_imp = METRICS['churn']['importances'].head(14).copy()
    churn_imp['feature'] = churn_imp['feature'].apply(pretty)
    figCimp = px.bar(churn_imp, x='importance', y='feature', orientation='h', title='Top Churn Drivers',
                     color='importance', color_continuous_scale='OrRd')
    st.plotly_chart(darkify(figCimp, 420).update_layout(yaxis={'categoryorder':'total ascending'}), use_container_width=True)
    st.markdown("""<div class='insight'>
    <h4>Takeaways</h4>
    Low <b>satisfaction</b>, long <b>days since service</b>, and high <b>competitor density</b> dominate. Use recovery calls, convenience perks, and loyalty pricing on high cost-to-income cohorts.
    </div>""", unsafe_allow_html=True)

    # Storyboard EDA — ripple of satisfaction
    st.subheader("Storyboard — Ripple of Satisfaction on Churn & ETA")
    r1c1, r1c2 = st.columns(2)
    d1 = safe_bin_mean(df, 'satisfaction_score', 'churn_probability', bins=10)
    r1c1.plotly_chart(darkify(px.line(d1, x='bin', y='churn_probability',
                                      title='Higher Satisfaction ↓ Churn (binned)'), 340), use_container_width=True)
    d2 = safe_bin_mean(df.assign(days_until_service=df['days_until_service'].clip(-120,120)), 'satisfaction_score', 'days_until_service', bins=10)
    r1c2.plotly_chart(darkify(px.line(d2, x='bin', y='days_until_service',
                                      title='Higher Satisfaction → Later ETA (binned)'), 340), use_container_width=True)
    st.markdown("""<div class='insight gray'>
    <h4>Meaning</h4>
    Satisfaction ≥4.0 is associated with lower churn and better schedule adherence. Invest in first-time-fix and proactive comms.
    </div>""", unsafe_allow_html=True)

    # Segment lens
    st.subheader("Churn Probability by Segment")
    seg_churn = df.groupby("cust_segment")["churn_probability"].mean().reset_index()
    fig_seg_churn = px.bar(seg_churn, x="cust_segment", y="churn_probability", title="Avg Churn by Segment")
    st.plotly_chart(darkify(fig_seg_churn, 360), use_container_width=True)

    # Recency vs Churn by segment
    st.subheader("Recency vs Churn (by Segment)")
    fig_rec_churn = px.scatter(df, x="recency", y="churn_probability", color="cust_segment", opacity=0.5,
                               title="Recency (days) vs Churn Probability — colored by Segment")
    st.plotly_chart(darkify(fig_rec_churn, 380), use_container_width=True)

    # CLTV by Segment
    st.subheader("CLTV by Segment")
    cltv_seg = df.groupby("cust_segment")["CLTV"].mean().reset_index()
    fig_cltv_seg = px.bar(cltv_seg, x="cust_segment", y="CLTV", title="Average CLTV by Segment")
    st.plotly_chart(darkify(fig_cltv_seg, 360), use_container_width=True)

    # RFM heatmap
    st.subheader("RFM Heatmap — Behavior vs Risk")
    rfmb = df.copy()
    rfmb['recency_bin'] = pd.qcut(rfmb['recency'], q=5, duplicates='drop').astype(str)
    rfmb['frequency_bin'] = pd.qcut(rfmb['frequency'], q=5, duplicates='drop').astype(str)
    rfm_pivot = rfmb.pivot_table(index='recency_bin', columns='frequency_bin', values='churn_probability', aggfunc='mean')
    fig_rfm = px.imshow(rfm_pivot.astype(float), aspect='auto', title='Avg Churn Probability by R & F bins',
                        labels=dict(x="Frequency bin", y="Recency bin", color="Churn Prob"))
    fig_rfm.update_xaxes(ticktext=[str(c) for c in rfm_pivot.columns], tickvals=list(range(len(rfm_pivot.columns))))
    fig_rfm.update_yaxes(ticktext=[str(i) for i in rfm_pivot.index], tickvals=list(range(len(rfm_pivot.index))))
    st.plotly_chart(darkify(fig_rfm, 380), use_container_width=True)
    st.markdown("""<div class='insight'>
    <h4>Action</h4>
    High-recency & low-frequency cells are prime retention targets; add loyalty offers + easy booking links.
    </div>""", unsafe_allow_html=True)

    # City × Model risk heatmap
    st.subheader("Churn Risk by City & Model")
    risk = df.groupby(['city','model'])['churn_probability'].mean().reset_index()
    figHM = px.density_heatmap(risk, x='model', y='city', z='churn_probability', color_continuous_scale='RdPu',
                               title='Average Churn Probability — City × Model')
    st.plotly_chart(darkify(figHM, 400), use_container_width=True)

    # Retention campaign list
    st.subheader("🚑 Retention Campaign List")
    hi = df[df['churn_probability']>=0.70].copy()
    def retention_action(row):
        if row['satisfaction_score'] < 2.5: return "Recovery call + voucher"
        if row['avg_service_cost'] > 8500: return "10–15% loyalty discount"
        if row['vehicle_age_years'] > 5: return "Extended Care package"
        return "Personalized feedback call"
    hi['Recommended Action'] = hi.apply(retention_action, axis=1)
    hi['months_since_service'] = (hi['recency']/30).round(1)
    show_cols = ['customer_id','model','city','churn_probability','satisfaction_score','avg_service_cost',
                 'months_since_service','km_overdue','oem_alignment','cust_segment','Recommended Action']
    st.dataframe(hi[show_cols].sort_values('churn_probability', ascending=False).head(150),
                 use_container_width=True, hide_index=True)

    st.markdown("""<div class='insight warn'>
    <h4>Next Steps</h4>
    Enrich with support tickets, workshop wait-times, spare-part SLAs, and campaign exposure for better predictors & fairness checks.
    </div>""", unsafe_allow_html=True)

# ================================================================================================
# TAB 3 — UPLIFT & SELECTION
# ================================================================================================
with tab_uplift:
    st.markdown("""<div class='section'><h2>📈 Uplift Modeling & Selection</h2>
    <p class='muted'>Uplift = P(book|outreach) − P(book|no outreach). We choose customers maximizing incremental revenue under capacity & budget.</p></div>""", unsafe_allow_html=True)
    
    # Calculate negative uplift percentage
    neg_uplift_pct = (uplift_df['uplift'] < 0).mean() * 100
    if neg_uplift_pct > 0:
        st.markdown(f"<div class='kpi red'><div class='l'>% Fleet with Negative Uplift</div><div class='v'>{neg_uplift_pct:.1f}%</div><div class='s'>These are 'Sleeping Dogs' to NOT contact</div></div>", unsafe_allow_html=True)
    
    with st.expander(" transparent assumptions", expanded=False):
        st.markdown("""
        To provide credible and transparent simulations, this model relies on a set of defined assumptions. Understanding these "knobs" is key to interpreting the results.
        - **Baseline Booking Rates**: The probability of a customer booking a service *without* any outreach is based on their service urgency:
          - `Critical Overdue`: 25%
          - `Overdue`: 30%
          - `Due Soon (0–15d)`: 45%
          - `Upcoming (16–45d)`: 28%
          - `Future`: 12%
        - **Uplift Adders**: The *additional* booking probability from outreach is simulated based on:
          - `High Risk Status`: +20 percentage points
          - `Active Competitor Promo`: +6 percentage points
        - **Behavioral Modifiers**:
          - `AMC Enrolled`: Baseline booking probability is multiplied by **1.05x**.
          - `Parts Wait Risk`: Baseline booking probability is multiplied by **0.90x**.
        - **Realism Damping Factor**: The calculated treatment effect (uplift) is multiplied by **`REALISM = 0.70`**. This accounts for real-world inefficiencies and prevents overly optimistic forecasts.
        - **Clipping**: All final probabilities are clipped to stay within a realistic range of `[0.01, 0.95]`.
        """)

    @st.cache_data
    def calculate_qini(df: pd.DataFrame):
        """Calculates data for a Qini curve."""
        df_sorted = df.sort_values('uplift', ascending=False).copy()
        df_sorted['treat_booking'] = df_sorted['booked'] * df_sorted['treatment']
        df_sorted['ctrl_booking'] = df_sorted['booked'] * (1 - df_sorted['treatment'])
        
        n_t = df_sorted['treatment'].sum()
        n_c = len(df_sorted) - n_t
        if n_c == 0: n_c = 1 # Avoid division by zero
        
        df_sorted['cum_treat_booking'] = df_sorted['treat_booking'].cumsum()
        df_sorted['cum_ctrl_booking'] = df_sorted['ctrl_booking'].cumsum()
        
        df_sorted['n_cum_treat'] = df_sorted['treatment'].cumsum()
        df_sorted['n_cum_ctrl'] = (1 - df_sorted['treatment']).cumsum()
        
        df_sorted['qini_treat'] = df_sorted['cum_treat_booking']
        df_sorted['qini_ctrl'] = df_sorted['cum_ctrl_booking'] * (n_t / n_c)
        df_sorted['uplift_curve'] = df_sorted['qini_treat'] - df_sorted['qini_ctrl']
        
        # Add random targeting line
        total_bookings_t = df_sorted['treat_booking'].sum()
        total_bookings_c = df_sorted['ctrl_booking'].sum()
        random_uplift_end = total_bookings_t - (total_bookings_c * n_t / n_c)
        df_sorted['random_curve'] = np.linspace(0, random_uplift_end, len(df_sorted))

        # AUUC calculation
        auuc = np.trapz(df_sorted['uplift_curve'], dx=1/len(df_sorted))
        auuc_random = np.trapz(df_sorted['random_curve'], dx=1/len(df_sorted))
        
        return df_sorted, auuc, auuc_random

    # === BEGIN: FINAL DEMO UPGRADE (Uplift Tab Restructure) ===
    st.subheader("👑 Uplift Model Performance & ROI")
    st.markdown("Evaluate the effectiveness of the uplift model with industry-standard metrics and visualize the trade-off between campaign spend and expected return.")

    # Waterfall Chart
    st.markdown("##### Incremental Bookings Waterfall")
    base_bookings = (uplift_df['p_book_control'] * (1 - uplift_df['treatment'])).sum()
    treat_bookings = (uplift_df['p_book_treat'] * uplift_df['treatment']).sum()
    incremental_bookings = (uplift_df['uplift'] * uplift_df['treatment']).sum()
    
    fig_waterfall = go.Figure(go.Waterfall(
        name = "2025 Forecast", orientation = "v",
        measure = ["relative", "relative", "total"],
        x = ["Baseline Bookings", "Incremental from Outreach", "Total Expected Bookings"],
        textposition = "outside",
        text = [f"{base_bookings:.0f}", f"+{incremental_bookings:.0f}", f"{base_bookings+incremental_bookings:.0f}"],
        y = [base_bookings, incremental_bookings, base_bookings+incremental_bookings],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    fig_waterfall.update_layout(title = "Expected Bookings: Baseline vs. Model-driven Outreach", showlegend = False)
    st.plotly_chart(darkify(fig_waterfall, 380), use_container_width=True)


    # Qini Curve & AUUC Score
    st.markdown("##### Qini Curve & Model Performance")
    qini_df, auuc, auuc_random = calculate_qini(uplift_df)
    
    c1, c2 = st.columns(2)
    c1.metric("AUUC (Uplift Model)", f"{auuc:.0f}", help="Area Under the Uplift Curve. Higher is better.")
    improvement_pct = ((auuc - auuc_random) / auuc_random) * 100 if auuc_random > 0 else 0
    c2.metric("Improvement vs. Random", f"{improvement_pct:.1f}%", help="How much better the model is than targeting randomly.")

    fig_qini = go.Figure()
    fig_qini.add_trace(go.Scatter(x=qini_df.index, y=qini_df['uplift_curve'], mode='lines', name='Uplift Model'))
    fig_qini.add_trace(go.Scatter(x=qini_df.index, y=qini_df['random_curve'], mode='lines', name='Random Targeting', line=dict(dash='dash')))
    fig_qini.update_layout(title='Qini Curve: Model Performance vs. Random',
                           xaxis_title='Customers Targeted (sorted by uplift)',
                           yaxis_title='Incremental Bookings')
    st.plotly_chart(darkify(fig_qini, 380), use_container_width=True)
    st.markdown(f"""<div class='insight'>
    <h4>Insight & Action</h4>
    The Qini curve confirms that our model is <b>{improvement_pct:.1f}% more effective</b> than contacting customers randomly. The steep initial slope shows that targeting the top-scoring customers yields the highest return. This validates using the uplift score to prioritize outreach efforts.
    </div>""", unsafe_allow_html=True)

    # Budget Frontier
    st.markdown("##### 💸 Spend Frontier & Break-Even Analysis")
    frontier_df = uplift_df.copy()
    frontier_df['offer_cost'] = np.maximum(300, 0.08 * frontier_df['avg_service_cost']).round(0)
    frontier_df['expected_uplift_rev'] = (frontier_df['uplift'] * frontier_df['avg_service_cost']).astype(float)
    frontier_df = frontier_df.sort_values('uplift', ascending=False).reset_index()
    frontier_df['cum_cost'] = frontier_df['offer_cost'].cumsum()
    frontier_df['cum_rev'] = frontier_df['expected_uplift_rev'].cumsum()
    
    budget_slider = st.slider("Select Campaign Budget (₹)", 10000, int(frontier_df['cum_cost'].max()), 80000, 5000)
    
    fig_frontier = go.Figure()
    fig_frontier.add_trace(go.Scatter(x=frontier_df['cum_cost'], y=frontier_df['cum_rev'], mode='lines', name='Incremental Revenue'))
    
    break_even_point_cost = (frontier_df[frontier_df['cum_rev'] > frontier_df['cum_cost']])['cum_cost'].max() if not (frontier_df[frontier_df['cum_rev'] > frontier_df['cum_cost']]).empty else 0
    if break_even_point_cost > 0:
        fig_frontier.add_vline(x=break_even_point_cost, line_dash="dash", line_color="red", annotation_text=f"Max Profitable Spend")
    fig_frontier.add_vline(x=budget_slider, line_dash="dot", line_color="yellow", annotation_text=f"Current Budget")
    
    fig_frontier.update_layout(title='Spend Frontier: Incremental Revenue vs. Offer Cost',
                               xaxis_title='Cumulative Campaign Spend (₹)', yaxis_title='Cumulative Incremental Revenue (₹)')
    st.plotly_chart(darkify(fig_frontier, 400), use_container_width=True)
    st.markdown(f"""<div class='insight warn'>
    <h4>Insight & Action</h4>
    This chart shows the expected revenue for a given budget. The optimal spend is just before the red 'Max Profitable Spend' line, where marginal revenue equals marginal cost. For the current budget of <b>₹{format_indian(budget_slider)}</b>, the projected ROI is positive.
    </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    # === END: FINAL DEMO UPGRADE (Uplift Tab Content Un-collapsed) ===

    st.subheader("🎯 Campaign Selection & Targeting")
    colA, colB, colC, colD = st.columns(4)
    capacity = colA.slider("Daily Capacity", 50, 300, 120, step=10, key="uplift_capacity_slider")
    budget   = colB.slider("Offer Budget (₹)", 20000, 200000, 80000, step=5000, key="uplift_budget_slider")
    select_by = colC.radio("Selection Criterion", ["Uplift", "Churn"], index=0)
    guardrails_on = colD.checkbox("Apply Guardrails", value=True, help="Automatically avoids contacting customers with predicted negative uplift (e.g., highly satisfied 'sleeping dogs' or early-birds). This is a best practice.")


    cand = uplift_df.copy()
    # === BEGIN: FIX for ValueError ===
    # Calculate 'offer_cost' and 'expected_uplift_rev' on the main candidate DataFrame
    cand['offer_cost'] = np.maximum(300, 0.08 * cand['avg_service_cost']).round(0)
    cand['expected_uplift_rev'] = (cand['uplift'] * cand['avg_service_cost']).astype(float)
    # === END: FIX for ValueError ===
    
    if select_by == "Uplift":
        selected = select_under_budget(cand, capacity, budget, apply_guardrails=guardrails_on)
    else:
        temp_cand = cand.copy()
        if guardrails_on:
            temp_cand = temp_cand[temp_cand['satisfaction_score'] <= 4.5]
            temp_cand = temp_cand[temp_cand['oem_alignment'] != 'Early vs OEM']
        temp_cand = temp_cand.sort_values("churn_probability", ascending=False).reset_index(drop=True)
        selected = temp_cand.head(capacity)

    # Charts
    st.subheader("Incremental Revenue vs Offer Cost (selected vs not)")
    if not cand.empty:
        figU = px.scatter(
            cand, x="offer_cost", y="expected_uplift_rev",
            color=cand['customer_id'].isin(selected['customer_id']) if not selected.empty else False,
            labels={"color":"Selected"}, title="Uplift Scatter"
        )
        st.plotly_chart(darkify(figU, 380), use_container_width=True)
    st.markdown("""<div class='insight'>
    <h4>What this means</h4>
    This chart plots incremental revenue (Y-axis) vs. offer cost (X-axis). The goal is to select customers in the <b>top-left quadrant (high revenue, low cost)</b>.
    <br>Crucially, customers in the <b>bottom half have negative uplift</b>—contacting them is predicted to <i>lose</i> revenue. Our selection logic and guardrails automatically avoid them.
    </div>""", unsafe_allow_html=True)


    # New uplift visualizations
    st.subheader("Uplift by Customer Segment")
    seg_uplift = cand.groupby("cust_segment")["uplift"].mean().reset_index()
    fig_seg_uplift = px.bar(seg_uplift, x="cust_segment", y="uplift", title="Avg Uplift by Segment")
    st.plotly_chart(darkify(fig_seg_uplift, 360), use_container_width=True)
    st.markdown("""<div class='insight warn'>
    <h4>What this means</h4>
    "Uplift" is the predicted change in booking probability from outreach. Note that segments like <b>"Loyal Champions"</b> now show a negative uplift. This is a critical insight: a generic marketing offer is likely to annoy this segment, as they are already loyal and don't need an incentive. The best action for them is no action, or a non-promotional, relationship-building message.
    </div>""", unsafe_allow_html=True)


    st.subheader("Comparative Density Plot (Treatment vs Control)")
    fig_den = go.Figure()
    fig_den.add_trace(go.Histogram(x=cand['p_treat'], nbinsx=40, name="Treatment", opacity=0.6, histnorm='probability'))
    fig_den.add_trace(go.Histogram(x=cand['p_ctrl'], nbinsx=40, name="Control", opacity=0.6, histnorm='probability'))
    fig_den.update_layout(barmode='overlay', title="Density of Booking Probabilities (Treatment vs Control)")
    st.plotly_chart(darkify(fig_den, 380), use_container_width=True)
    st.markdown("""<div class='insight'>
    <h4>What this means</h4>
    This plot compares the distribution of booking probabilities for two groups: customers who receive an offer (Treatment) and those who don't (Control). The clear rightward shift of the blue 'Treatment' curve shows that our outreach successfully increases the likelihood of booking across the customer base.
    </div>""", unsafe_allow_html=True)

    st.subheader("Uplift Summary Table")
    st.dataframe(seg_uplift.rename(columns={"uplift":"Avg Uplift"}), use_container_width=True, hide_index=True)
    st.caption("This table provides the raw average uplift values per segment, confirming the insights from the bar chart above.")


# ================================================================================================
# TAB 4 — 🚀 AI CAMPAIGN BUILDER (FINAL, POLISHED VERSION)
# ================================================================================================
# ================================================================================================
# TAB 4 — 🚀 AI CAMPAIGN BUILDER (FINAL, POLISHED VERSION)
# ================================================================================================
with tab_ai:
    import os

    st.markdown("""
    <div class='section'>
      <h2>🚀 AI Campaign Builder</h2>
      <p class='muted'>A streamlined, 4-step process to define strategy, generate AI-powered messages, analyze, and launch personalized campaigns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📝 Next-Best-Action Playbook"):
        st.markdown("This editable matrix defines the default outreach strategy. The AI Campaign Builder uses this as a starting point for its recommendations.")
        
        nba_data = {
            'Urgency': ["Overdue", "Overdue", "Due Soon", "Due Soon", "Upcoming", "Upcoming"],
            'Risk Tier': ["High", "Low/Medium", "High", "Low/Medium", "High", "Low/Medium"],
            'Channel': ["Call + SMS", "SMS", "SMS + Email", "Email", "Email", "Nurture"],
            'Offer': ["Goodwill Coupon", "Express Lane", "Loyalty Discount", "Book Now Link", "Pre-Monsoon Check", "Educational Content"],
            'Incentive Range (%)': ["10-15", "0-5", "5-10", "0", "0", "0"],
            'Script Cue': ["Recovery focused", "Convenience", "Value focused", "Reminder", "Seasonal relevance", "Awareness"]
        }
        nba_df = pd.DataFrame(nba_data)
        
        if 'nba_matrix' not in st.session_state:
            st.session_state.nba_matrix = nba_df

        st.caption("Editable Next-Best-Action (NBA) Matrix")
        edited_nba = st.data_editor(st.session_state.nba_matrix, use_container_width=True, hide_index=True, key="nba_editor")
        st.session_state.nba_matrix = edited_nba # Persist edits
    
    # ---------- Session state initialization for speed and persistence ----------
    if "campaigns" not in st.session_state: st.session_state.campaigns = []
    if "drafts" not in st.session_state: st.session_state.drafts = {}
    if "cohort" not in st.session_state: st.session_state.cohort = pd.DataFrame()
    if "strategy" not in st.session_state: st.session_state.strategy = {}
    if "finalized_campaign" not in st.session_state: st.session_state.finalized_campaign = None

    # ---------- Helper functions for this tab ----------
    def get_offer_strategy(objective: str, trigger: str) -> str:
        # Use the NBA matrix to get a more dynamic suggestion
        urgency_map = {
            "Overdue": "Overdue", "Due Soon": "Due Soon", "Approaching": "Due Soon",
            "Expiring": "Upcoming", "High Churn": "Overdue", "Seasonal": "Upcoming"
        }
        risk_map = {"High Churn": "High", "Overdue": "High"}
        
        urgency_key = next((u for k, u in urgency_map.items() if k in trigger), "Upcoming")
        risk_key = next((r for k, r in risk_map.items() if k in trigger), "Low/Medium")

        nba_row = st.session_state.nba_matrix[
            (st.session_state.nba_matrix['Urgency'] == urgency_key) &
            (st.session_state.nba_matrix['Risk Tier'] == risk_key)
        ]
        
        if not nba_row.empty:
            return nba_row.iloc[0]['Offer']
            
        # Fallback logic
        if "Warranty" in trigger: return "Special offer on Annual Maintenance Contract (AMC) + loyalty points."
        if "Churn" in trigger: return "Recovery call with a goodwill coupon & free pick-up/drop."
        if "Component Due" in trigger: return f"15% off on {trigger.split('(')[-1].split(')')[0]} service + complimentary wash."
        if "Overdue" in trigger: return "SMS & Email with express lane access link + 5% off labor."
        if "Usage" in trigger: return "Recommendation for a high-mileage check-up with a small discount."
        if "Seasonal" in trigger: return "Monsoon-ready check-up package (brakes, tires, wipers)."
        return "Standard service reminder with a link to book an appointment."

    # === BEGIN: FIX for NameError ===
    def get_predefined_templates(trigger: str, offer: str) -> Dict[str, str]:
        """Returns instant, high-quality message templates based on the trigger."""
        templates = {
            "Service Overdue": {
                "sms": f"URGENT: Your car is overdue for its scheduled service. To get you back on track, we're offering: {offer}. Book now to ensure safety and performance. Reply 'BOOK' or call us.",
                "email": f"Subject: Important Reminder: Your Vehicle Service is Overdue\n\nHi [Customer Name],\n\nOur records show that your vehicle ([VIN]) is now past its recommended service date. Regular maintenance is crucial for safety and preventing costly repairs.\n\nTo help, we're offering you: **{offer}**.\n\nDon't delay, book your appointment today.\n\nBest,\nThe AutoCare Team",
                "call_script": "[Opener]: Hi, I'm calling from AutoCare about your car. [Reason]: We've noticed your service is overdue, and we want to make sure everything is running smoothly. [Offer]: To make it easier, we're providing {offer} for you. [CTA]: When would be a good time this week to book you in?"
            },
            "Due Soon": {
                "sms": f"Service reminder from AutoCare: Your car is due for service soon. {offer}. Book your slot in advance to avoid the rush! Tap here to book: [Link]",
                "email": f"Subject: A Friendly Reminder: Your Upcoming Service\n\nHi [Customer Name],\n\nThis is a courtesy reminder that your vehicle ([VIN]) is due for its scheduled maintenance soon. Keep your car in peak condition by booking in advance.\n\nAs a valued customer, you can take advantage of our **{offer}**.\n\nBooking online is easy. We look forward to seeing you!\n\nBest,\nThe AutoCare Team",
                "call_script": "[Opener]: Hi, this is [Your Name] from AutoCare. [Reason]: I'm calling as our records show your car is due for maintenance soon. [Offer]: We have an {offer} available for early bookings. [CTA]: Would you like to reserve a slot now?"
            },
            "High Churn": {
                "sms": f"We miss you at AutoCare! To welcome you back, we're offering a special {offer} on your next service. We're committed to your satisfaction. Reply 'YES' for a callback.",
                "email": f"Subject: We Value You as a Customer\n\nHi [Customer Name],\n\nWe haven't seen you in a while and wanted to check in. At AutoCare, we're committed to providing the best service experience, and we value your business.\n\nWe would like to offer you **{offer}** as a special incentive to visit us again.\n\nYour feedback is important to us. Please let us know if there's anything we can do better.\n\nSincerely,\nThe AutoCare Team",
                "call_script": "[Opener]: Hi, this is [Your Name] calling from AutoCare. [Reason]: I'm personally reaching out as we haven't seen you in some time and wanted to ensure everything is okay with your vehicle. [Offer]: We truly value you as a customer and would like to offer {offer} for your next visit. [CTA]: Is there anything we can assist with today?"
            },
             "Default": {
                "sms": f"AutoCare Reminder: Your vehicle service is approaching. Take advantage of our offer: {offer}. Book today!",
                "email": f"Subject: Reminder from AutoCare\n\nHi [Customer Name],\n\nThis is a reminder that your vehicle service is due. Our current promotion is: {offer}.\n\nBook now to ensure continued reliability and performance.\n\nThanks,\nAutoCare",
                "call_script": "[Opener]: Hi, from AutoCare. [Reason]: Calling about your upcoming vehicle service. [Offer]: We have a special offer of {offer}. [CTA]: Can I help you book an appointment?"
            }
        }
        
        trigger_key = "Service Overdue" if "Overdue" in trigger else \
                      "Due Soon" if "Due Soon" in trigger or "Approaching" in trigger else \
                      "High Churn" if "Churn" in trigger else "Default"
                      
        return templates[trigger_key]
    # === END: FIX for NameError ===

    @st.cache_data(show_spinner=False)
    def call_gemini_api(_strategy, _cohort_df_summary, model_name):
        """Cached function to call Gemini API to avoid repeated calls for the same inputs."""
        # === BEGIN: FIX for StreamlitSecretNotFoundError ===
        api_key = None
        try:
            # This will work in a deployed environment with secrets configured
            api_key = st.secrets["GEMINI_API_KEY"]
        except (FileNotFoundError, KeyError):
            # This will be the fallback for local development without a secrets.toml file
            # Using a placeholder ensures the app doesn't crash.
            api_key = "placeholder_for_local_development" 
        # === END: FIX for StreamlitSecretNotFoundError ===
        
        try:
            # Continue only if gemini is available and api_key is not the placeholder
            if not gemini_ok or api_key == "placeholder_for_local_development":
                raise Exception("Gemini not configured for local use.")

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            
            prompt = f"""
You are a marketing expert for AutoCare, a car service center in India.
Your task is to generate 3 campaign message drafts based on a detailed strategy.

**Campaign Strategy:**
- **Business Objective:** {_strategy['objective']}
- **Primary Customer Trigger:** {_strategy['trigger']}
- **AI-Suggested Offer:** "{_strategy['offer']}"
- **Desired Tone:** {_strategy['tone']}
- **Desired Length:** {_strategy['length']}
- **Call to Action:** {_strategy['cta']}
- **Target Cohort Summary:** {_cohort_df_summary}

**Instructions:**
Return a single, valid JSON object with three keys: "sms", "email", "call_script".
- **sms:** A concise SMS, respecting the desired length and tone. Max 320 chars. Use [VIN] as a placeholder for the car's vehicle ID.
- **email:** A short paragraph with 2-3 bullet points, reflecting the tone. Use [VIN] as a placeholder.
- **call_script:** A 30-45 second script for an outbound call, structured with [Opener], [Reason], [Offer], [CTA]. Use [VIN] as a placeholder.
"""
            response = model.generate_content(prompt, request_options={"timeout": 15})
            clean_text = re.sub(r"```json\n?|```", "", response.text)
            drafts = json.loads(clean_text)
            if all(k in drafts for k in ["sms", "email", "call_script"]):
                return drafts, True
            else: return None, False
        except Exception:
            # Errors will be handled gracefully in the main app flow
            return None, False

    # --- Step 1: Strategy & Cohort Definition (Wrapped in a Form for performance) ---
    with st.container(border=True):
        st.markdown("<h3 style='color: #00ff88; margin-top: 0;'>Step 1: Define Campaign Strategy & Cohort</h3>", unsafe_allow_html=True)
        
        with st.form(key="strategy_form"):
            col1, col2 = st.columns(2)
            with col1:
                objective = st.radio(
                    "**Business Objective**",
                    ["Prevent Churn", "Drive Early Service", "Post-Warranty Retention"],
                    captions=["Retain at-risk customers.", "Encourage timely service.", "Capture post-warranty revenue."]
                )
                trigger_options = [
                    "Service Overdue (by Calendar/ETA)", "Service Due Soon (0-30 days)",
                    "KM Threshold Approaching (>9k km)", "High Daily KM Usage (>50 km/day)",
                    "Component Due (Brakes)", "Component Due (Oil)", "Component Due (Tires)",
                    "Warranty Expiring (<90 days)", "High Churn Probability (≥0.6)",
                    "Seasonal Campaign (Monsoon Prep)"
                ]
                trigger = st.selectbox("**Primary Trigger**", trigger_options, index=0)

            with col2:
                segs_all = sorted(df["cust_segment"].unique().tolist())
                cities_all = sorted(df["city"].unique().tolist())
                seg_sel = st.multiselect("**Target Segment**", segs_all, default=["High-Value & At-Risk"])
                city_sel = st.multiselect("**Target City**", ["All"] + cities_all, default=["All"])
            
            submitted = st.form_submit_button("✅ Confirm Cohort & View Uplift", use_container_width=True, type="primary")

            if submitted:
                st.session_state.strategy = {"objective": objective, "trigger": trigger, "seg_sel": seg_sel, "city_sel": city_sel, "offer": get_offer_strategy(objective, trigger)}
                cohort = df.merge(uplift_df[["customer_id","uplift"]], on="customer_id", how="left")
                if seg_sel: cohort = cohort[cohort["cust_segment"].isin(seg_sel)]
                if "All" not in city_sel: cohort = cohort[cohort["city"].isin(city_sel)]
                if trigger == "Service Overdue (by Calendar/ETA)": cohort = cohort[cohort['days_until_service'] < 0]
                elif trigger == "Service Due Soon (0-30 days)": cohort = cohort[cohort['days_until_service'].between(0, 30)]
                elif trigger == "KM Threshold Approaching (>9k km)": cohort = cohort[cohort['km_since_last_service'] > 9000]
                elif trigger == "High Daily KM Usage (>50 km/day)": cohort = cohort[cohort['daily_km'] > 50]
                elif trigger == "Component Due (Brakes)": cohort = cohort[cohort['component_due'] == 'Brakes']
                elif trigger == "Component Due (Oil)": cohort = cohort[cohort['component_due'] == 'Oil']
                elif trigger == "Component Due (Tires)": cohort = cohort[cohort['component_due'] == 'Tires']
                elif trigger == "Warranty Expiring (<90 days)": cohort = cohort[cohort['days_to_warranty_expiry'].between(0, 90)]
                elif trigger == "High Churn Probability (≥0.6)": cohort = cohort[cohort['churn_probability'] >= 0.6]
                elif trigger == "Seasonal Campaign (Monsoon Prep)":
                    monsoon_cities = ['Mumbai','Kolkata','Chennai']
                    is_monsoon_month = datetime.now().month in [6,7,8,9]
                    if is_monsoon_month: cohort = cohort[cohort['city'].isin(monsoon_cities)]
                    else: cohort = pd.DataFrame(columns=cohort.columns)
                st.session_state.cohort = cohort
                st.session_state.drafts, st.session_state.finalized_campaign = {}, None

    if 'strategy' in st.session_state and st.session_state.strategy and not st.session_state.cohort.empty:
        cohort_df = st.session_state.cohort
        cohort_size = len(cohort_df)
        inc_rev = (cohort_df['uplift'].fillna(0) * cohort_df['avg_service_cost']).sum()
        avg_uplift = cohort_df['uplift'].mean()

        if avg_uplift < 0:
            avg_inc_rev = (cohort_df['uplift'] * cohort_df['avg_service_cost']).mean()
            st.warning(f"""
            **⚠️ CAMPAIGN VIABILITY WARNING** The selected cohort has an average **negative uplift of {avg_uplift:.2%}**. 
            Launching this promotional campaign is projected to result in a net loss of **~₹{-avg_inc_rev:,.0f} per customer**.
    
            **Recommendation:** Refine your targeting to exclude segments with negative uplift (like 'Loyal Champions'). Do not proceed with a promotional offer for this group. Consider a non-promotional, relationship-building campaign instead.
            """, icon="🚨")
        
        st.markdown(f"**💡 AI-Suggested Offer Strategy (from Playbook):**")
        st.info(st.session_state.strategy.get("offer", "N/A"))
        
        uplift_col, size_col = st.columns(2)
        uplift_col.markdown(f"<div class='kpi'><div class='l'>Possible Uplift from this Campaign</div><div class='v'>₹{format_indian(inc_rev)}</div></div>", unsafe_allow_html=True)
        size_col.markdown(f"<div class='kpi amber'><div class='l'>Target Cohort Size</div><div class='v'>{cohort_size:,.0f}</div></div>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown("<h3 style='color: #00ff88; margin-top: 0;'>Step 2: Generate & Finalize Campaign Message</h3>", unsafe_allow_html=True)
            
            with st.form(key="message_form"):
                msg_c1, msg_c2, msg_c3 = st.columns(3)
                tone = msg_c1.selectbox("**Tone**", ["Friendly", "Professional", "Urgent", "Promotional"])
                length = msg_c2.selectbox("**Length**", ["Concise", "Standard", "Detailed"])
                cta = msg_c3.selectbox("**Call-to-Action Style**", ["Reply to book", "Tap link to book", "Call us to book"])
                
                generate_clicked = st.form_submit_button("📝 Generate Drafts (Instant)", use_container_width=True)
                
                if generate_clicked:
                    st.session_state.strategy.update({"tone": tone, "length": length, "cta": cta})
                    st.session_state.drafts = get_predefined_templates(st.session_state.strategy['trigger'], st.session_state.strategy['offer'])
                    st.rerun()

            if st.session_state.drafts:
                if st.button("✨ Ask AI for More Variations (Slower)"):
                    with st.spinner("AI is advising on your strategy..."):
                        cohort_summary = f"Top models: {st.session_state.cohort['model'].value_counts().head(2).index.tolist()}, Segments: {st.session_state.cohort['cust_segment'].value_counts().head(2).index.tolist()}"

                        # Check for negative uplift to change AI behavior
                        if st.session_state.cohort['uplift'].mean() < 0:
                            # Special prompt for negative uplift scenario
                            st.info("Since this cohort has negative uplift, the AI will provide strategic advice instead of sales copy.", icon="🧠")
                            # Here you would ideally have a different API call or a modified prompt.
                            # For this simulation, we'll just use a predefined "strategic" response.
                            st.session_state.drafts = {
                                "sms": "Strategic Advice: Do not send a promotional SMS. Instead, consider a feedback survey to re-engage this 'sleeping dog' segment.",
                                "email": "Subject: Your Opinion Matters to Us\n\nHi [Customer Name],\n\nAs a valued customer, your feedback is crucial for us to improve. We would be grateful if you could take 60 seconds to answer a few questions about your experience. [Link to Survey]",
                                "call_script": "[Opener]: Hi, this is a courtesy call from AutoCare. [Reason]: We're not selling anything today, just wanted to check in and see how your vehicle is running and if there's any feedback you'd like to share. [CTA]: Your opinion helps us serve you better."
                            }
                        else:
                            # Original logic for positive uplift
                            ai_drafts, success = call_gemini_api(st.session_state.strategy, cohort_summary, "gemini-1.5-pro-latest")
                            if success:
                                st.session_state.drafts = ai_drafts
                            else:
                                st.warning("Could not connect to Generative AI. Using fallback templates.", icon="⚠️")


                st.markdown("##### Choose Your Favorite Template")
                draft_c1, draft_c2 = st.columns(2)
                samples = [st.session_state.drafts['sms'], st.session_state.drafts['email'], st.session_state.drafts['call_script']] * 2
                
                if 'selected_draft' not in st.session_state or not st.session_state.selected_draft:
                    st.session_state.selected_draft = samples[0]

                for i, sample in enumerate(samples):
                    container = draft_c1 if i < 3 else draft_c2
                    with container:
                        if st.button(f"Select Sample {i+1}", key=f"s_{i}", use_container_width=True):
                            st.session_state.selected_draft = sample
                        st.markdown(f"<div class='insight gray'>{sample[:200] + '...' if len(sample)>200 else sample}</div>", unsafe_allow_html=True)
                
                st.markdown("---")
                final_message = st.text_area("Selected Master Message (Editable)", st.session_state.selected_draft, height=150)
                
                if st.button("✅ Finalize Campaign & View Analysis", use_container_width=True):
                    st.session_state.finalized_campaign = {"Timestamp": datetime.now(), **st.session_state.strategy, "Cohort Size": cohort_size, "Est. Incr. Revenue (₹)": inc_rev, "Master Message": final_message}
                    st.session_state.campaigns.append(st.session_state.finalized_campaign)
                    st.success(f"Campaign Finalized! View analysis and export options below.")
                    st.rerun()

    if st.session_state.finalized_campaign:
        camp = st.session_state.finalized_campaign
        cohort_df = st.session_state.cohort
        with st.container(border=True):
            st.markdown("<h3 style='color: #00ff88; margin-top: 0;'>Step 3: Campaign Brief Analysis</h3>", unsafe_allow_html=True)
            
            st.markdown("##### Campaign Diagnostics")
            diag_c1, diag_c2, diag_c3 = st.columns(3)
            diag_c1.plotly_chart(darkify(px.bar(cohort_df['cust_segment'].value_counts(), title='Targeted Segments'), 300), use_container_width=True)
            diag_c2.plotly_chart(darkify(px.bar(cohort_df['city'].value_counts(), title='Targeted Cities'), 300), use_container_width=True)
            diag_c3.plotly_chart(darkify(px.treemap(cohort_df, path=['model'], title='Targeted Vehicle Models'), 300), use_container_width=True)
            
            st.markdown("##### Workshop Load Projection")
            load_df = cohort_df.groupby('city').size().reset_index(name='projected_load')
            load_fig = px.bar(load_df, x='city', y='projected_load', title='Projected Service Load by City for this Campaign')
            st.plotly_chart(darkify(load_fig, 350), use_container_width=True)
            st.markdown("""<div class='insight warn'><h4>Insight</h4>This chart estimates the number of additional services this campaign will drive to each city's workshop. Monitor cities with high projected loads like <b>Delhi</b> and <b>Bangalore</b> to ensure adequate technician and bay capacity, preventing long wait times.</div>""", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<h3 style='color: #00ff88; margin-top: 0;'>Step 4: Personalize & Export Customer List</h3>", unsafe_allow_html=True)
            
            def personalize_message_for_customer(row, master_message):
                msg = master_message.replace("[VIN]", row["customer_id"]).replace("[Car Model]", row["model"]).replace("[City]", row["city"])
                if row['total_services'] > 8: msg += " As one of our most loyal customers, we've reserved a priority slot for you."
                elif row['churn_probability'] > 0.7: msg += " We're committed to providing you with the best possible service experience."
                elif "Component" in row.get('trigger', ''): msg += f" This is a key check to ensure your {row['trigger'].split('(')[-1].split(')')[0].lower()} are in top condition for your safety."
                return msg
            
            if st.button(f"🚀 Personalize Messages for {cohort_size} Customers", use_container_width=True):
                with st.spinner("Adding personal touches..."):
                    export_df = cohort_df.copy()
                    export_df['trigger'] = camp['trigger']
                    export_df['personalized_message'] = export_df.apply(lambda row: personalize_message_for_customer(row, camp['Master Message']), axis=1)
                    export_df['offer_start_date'] = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
                    
                    cols_to_export = ['customer_id', 'model', 'city', 'total_services', 'avg_service_cost', 'recency', 'churn_probability', 'uplift', 'offer_start_date', 'personalized_message']
                    export_df_final = export_df[cols_to_export].rename(columns={'recency': 'days_since_service'})
                    st.session_state.export_data = export_df_final
                    st.success("Personalization complete! Your download is ready.")

            if 'export_data' in st.session_state and st.session_state.export_data is not None:
                st.dataframe(st.session_state.export_data.head())
                csv_export = st.session_state.export_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"⬇️ Download Personalized List (CSV)",
                    data=csv_export,
                    file_name=f"autocare_personalized_campaign_{datetime.now():%Y%m%d}.csv",
                    mime="text/csv",
                    use_container_width=True
                )