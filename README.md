# AutoCare AI — ETA, Churn, Uplift & Outreach (Streamlit)

A production-style **CRM demo for automotive after-sales service**. The app simulates a customer fleet and demonstrates:

- **⏱️ ETA Intelligence & Scheduling** — hybrid ETA (calendar ⟂ km) aligned to OEM intervals, with empirical prediction bands and capacity/adherence analytics.
- **⚠️ Churn & RFM** — calibrated churn, RFM scoring, silent-churn detection using persona-based long-overdue thresholds.
- **📈 Uplift & Selection** — T-Learner (treatment/control heads) with realistic positive/negative uplift effects and “sleeping dogs” protection.
- **🚀 AI Campaign Builder** — message drafting with *Gemini* (optional), segment filters, guardrails, and exportable campaign assets.

> This README reflects the shipped app code. See `autocare_complete.py` for implementation. :contentReference[oaicite:1]{index=1}

---

## Key Concepts & What the Demo Shows

### 1) Data Design (Synthetic but business-realistic)
- **Fleet & usage:** model, city, vehicle age, last service recency; **OEM intervals** for km & days.
- **Hybrid due date:** `days_until_service = min(days_until_calendar, days_until_km) / weather_factor`.
- **Personas**: `{Light, Typical, Heavy, Commercial}` drive **annual_km** and **long_overdue_threshold**.
- **Churn-related fields:** satisfaction (CSI, NPS bucket), competitor promo, cost context, feedback reason.
- **Ops reality:** service type (Routine/Comprehensive/Repair), ECT hours, delays from parts/tech/insurance.

### 2) Features & Segments
- **RFM-style features**: `recency`, `frequency`, `monetary` + normalized `rfm_score`.
- **Customer segments** used across tabs:
  - **Loyal Champions**
  - **High-Value & At-Risk**
  - **New & Promising**
  - **Hibernating**

### 3) Models
- **ETA Regressor:** XGBoost regressor *(falls back to HistGradientBoostingRegressor if xgboost not installed)*.
- **Churn Classifier (calibrated):** XGBoost / RandomForest wrapped with `CalibratedClassifierCV` for probability fidelity.
- **Uplift (T-Learner):** Two heads (treatment vs. control) to estimate `p_treat - p_control` at a customer level with realism dampening and negative uplift in cases like over-contact or very recent visits.

### 4) Explainability & Ops
- **Empirical prediction intervals** for ETA via residual quantiles.
- **Why-this-ETA chips** combining service type and delay drivers.
- **Why-this-churn chips** mixing recency-threshold, CSI/NPS, and top features.
- **Scenario toggles** (Monsoon Spike, Monday Rush, Competitor Launch) shift capacity, no-shows, and delays to mimic real conditions.

---

## File Structure

