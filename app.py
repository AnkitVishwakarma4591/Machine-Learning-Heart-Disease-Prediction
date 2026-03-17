import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title="CHD Risk Predictor", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0e1a !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(180,30,60,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(220,60,40,0.12) 0%, transparent 55%),
        linear-gradient(160deg, #0a0e1a 0%, #0f1525 50%, #0a0e1a 100%) !important;
    min-height: 100vh;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { max-width: 800px !important; padding: 2.5rem 2rem 4rem !important; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.4rem 1rem 1.8rem;
}
.hero-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 70px; height: 70px;
    background: linear-gradient(135deg, #c0392b, #e74c3c);
    border-radius: 20px;
    margin: 0 auto 1.3rem;
    box-shadow: 0 8px 32px rgba(231,76,60,0.45);
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.5rem;
    color: #f5f0eb;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}
.hero p { color: #8a94a8; font-size: 0.95rem; font-weight: 300; }

/* ── Divider ── */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(231,76,60,0.4), rgba(192,57,43,0.6), rgba(231,76,60,0.4), transparent);
    margin: 1.6rem 0;
}

/* ── Section label ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #c0392b;
    margin: 1.6rem 0 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(192,57,43,0.22);
}

/* ── Streamlit widget overrides ── */
label,
.stSelectbox label,
.stNumberInput label {
    color: #b0bac8 !important;
    font-size: 0.81rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    margin-bottom: 0.25rem !important;
}

[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #f0eae4 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: rgba(192,57,43,0.6) !important;
    box-shadow: 0 0 0 3px rgba(192,57,43,0.15) !important;
    outline: none !important;
}
[data-testid="stSelectbox"] svg { color: #8a94a8 !important; }
[data-testid="stNumberInput"] button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #8a94a8 !important;
    border-radius: 8px !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #c0392b, #e74c3c) !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    height: auto !important;
    box-shadow: 0 4px 20px rgba(192,57,43,0.4) !important;
    transition: box-shadow 0.3s, transform 0.15s !important;
}
.stButton > button:hover {
    box-shadow: 0 8px 32px rgba(192,57,43,0.55) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result boxes ── */
.result-box {
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-top: 1.4rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    animation: slideUp 0.4s cubic-bezier(0.16,1,0.3,1);
}
.result-high {
    background: linear-gradient(135deg, rgba(192,57,43,0.18), rgba(231,76,60,0.1));
    border: 1px solid rgba(231,76,60,0.35);
    box-shadow: 0 4px 24px rgba(192,57,43,0.2);
}
.result-low {
    background: linear-gradient(135deg, rgba(39,174,96,0.15), rgba(46,204,113,0.08));
    border: 1px solid rgba(46,204,113,0.3);
    box-shadow: 0 4px 24px rgba(39,174,96,0.15);
}
.result-icon { flex-shrink: 0; margin-top: 2px; }
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    margin-bottom: 0.3rem;
}
.result-high .result-title { color: #e74c3c; }
.result-low  .result-title { color: #2ecc71; }
.result-desc { color: #8a94a8; font-size: 0.87rem; line-height: 1.55; }

/* ── Probability bar ── */
.prob-wrap { margin-top: 1.1rem; }
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #8a94a8;
    margin-bottom: 0.4rem;
    font-weight: 500;
}
.prob-track {
    height: 7px;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    overflow: hidden;
}
.prob-fill { height: 100%; border-radius: 99px; }
.prob-high { background: linear-gradient(90deg, #c0392b, #e74c3c); }
.prob-low  { background: linear-gradient(90deg, #27ae60, #2ecc71); }

/* ── Disclaimer ── */
.disclaimer {
    margin-top: 1.4rem;
    padding: 1rem 1.2rem;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    display: flex;
    gap: 0.7rem;
    align-items: flex-start;
}
.disclaimer p { color: #6b7585; font-size: 0.77rem; line-height: 1.55; }

@keyframes slideUp {
    from { opacity:0; transform:translateY(14px); }
    to   { opacity:1; transform:translateY(0); }
}

#MainMenu, footer, [data-testid="stStatusWidget"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")

loaded_model = load_model()

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">
    <svg width="34" height="34" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 21.593c-5.63-5.539-11-10.297-11-14.402C1 3.772 3.772 1 7.191 1c1.947 0 3.735.995 4.809 2.5C13.074 1.995 14.862 1 16.809 1 20.228 1 23 3.772 23 7.191c0 4.105-5.37 8.863-11 14.402z"
        fill="rgba(255,255,255,0.95)"/>
    </svg>
  </div>
  <h1>Coronary Heart Disease<br>Risk Predictor</h1>
  <p>Assess your 10-year CHD risk based on clinical &amp; lifestyle indicators</p>
</div>
<div class="styled-divider"></div>
""", unsafe_allow_html=True)

# ── Personal Information ──────────────────────────────────────────────────────
st.markdown('<div class="section-label">Personal Information</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=40)
with col2:
    sex = st.selectbox("Sex", ("Male", "Female"))
with col3:
    bmi = st.number_input("BMI", min_value=1.0, max_value=70.0, value=25.0)

# ── Smoking Habits ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Smoking Habits</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    current_smoker = st.selectbox("Current Smoker", ("No", "Yes"))
with col2:
    cigsPerDay = st.number_input("Cigarettes per Day", min_value=0, max_value=100, value=0)

# ── Cardiovascular Measurements ───────────────────────────────────────────────
st.markdown('<div class="section-label">Cardiovascular Measurements</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    heart_rate = st.number_input("Heart Rate (BPM)", min_value=1, max_value=250, value=75)
with col2:
    sysBP = st.number_input("Systolic BP", min_value=1.0, max_value=300.0, value=120.0)
with col3:
    diaBP = st.number_input("Diastolic BP", min_value=1.0, max_value=200.0, value=80.0)

# ── Blood & Metabolic Panel ───────────────────────────────────────────────────
st.markdown('<div class="section-label">Blood &amp; Metabolic Panel</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    totChol = st.number_input("Total Cholesterol", min_value=1.0, max_value=700.0, value=200.0)
with col2:
    glucose = st.number_input("Glucose Level", min_value=1.0, max_value=500.0, value=80.0)

# ── Medical History ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Medical History</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    BPMeds = st.selectbox("BP Medication", ("No", "Yes"))
with col2:
    prevalentStroke = st.selectbox("Prior Stroke", ("No", "Yes"))
with col3:
    prevalentHyp = st.selectbox("Hypertensive", ("No", "Yes"))
with col4:
    diabetes = st.selectbox("Diabetes", ("No", "Yes"))

st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
X_new = np.array([[
    age / 75,
    1 if sex == "Male" else 0,
    bmi,
    1 if current_smoker == "Yes" else 0,
    heart_rate, sysBP, diaBP, totChol, cigsPerDay,
    1 if BPMeds == "Yes" else 0,
    1 if prevalentStroke == "Yes" else 0,
    1 if prevalentHyp == "Yes" else 0,
    1 if diabetes == "Yes" else 0,
    glucose,
]])

if st.button("PREDICT MY RISK"):
    prediction  = loaded_model.predict(X_new)[0]
    probability = loaded_model.predict_proba(X_new)[0][1] * 100

    if prediction == 1:
        st.markdown(f"""
        <div class="result-box result-high">
          <div class="result-icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="12" cy="12" r="11" stroke="#e74c3c" stroke-width="1.5"/>
              <path d="M12 7v6" stroke="#e74c3c" stroke-width="2" stroke-linecap="round"/>
              <circle cx="12" cy="16.5" r="1" fill="#e74c3c"/>
            </svg>
          </div>
          <div style="flex:1">
            <div class="result-title">High Risk Detected</div>
            <div class="result-desc">Based on the indicators provided, there is an elevated probability of developing coronary heart disease within the next 10 years. Please consult a cardiologist.</div>
            <div class="prob-wrap">
              <div class="prob-label"><span>CHD Risk Probability</span><span>{probability:.1f}%</span></div>
              <div class="prob-track"><div class="prob-fill prob-high" style="width:{probability:.1f}%"></div></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box result-low">
          <div class="result-icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="12" cy="12" r="11" stroke="#2ecc71" stroke-width="1.5"/>
              <path d="M7.5 12.5l3 3 6-6" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
          <div style="flex:1">
            <div class="result-title">Low Risk</div>
            <div class="result-desc">Your current indicators suggest a low likelihood of developing coronary heart disease within the next 10 years. Maintain a healthy lifestyle.</div>
            <div class="prob-wrap">
              <div class="prob-label"><span>CHD Risk Probability</span><span>{probability:.1f}%</span></div>
              <div class="prob-track"><div class="prob-fill prob-low" style="width:{probability:.1f}%"></div></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" style="flex-shrink:0;margin-top:2px" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="11" stroke="#4a5568" stroke-width="1.5"/>
        <path d="M12 8v5" stroke="#4a5568" stroke-width="2" stroke-linecap="round"/>
        <circle cx="12" cy="15.5" r="1" fill="#4a5568"/>
      </svg>
      <p>This tool is for informational purposes only and does not constitute medical advice, diagnosis, or treatment. Always seek guidance from a qualified healthcare provider.</p>
    </div>
    """, unsafe_allow_html=True)