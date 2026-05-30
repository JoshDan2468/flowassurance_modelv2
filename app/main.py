from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
from datetime import datetime
import numpy as np # Added numpy for calculations in feature engineering

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# Plot
import matplotlib.pyplot as plt

# FastAPI Static Files
from fastapi.staticfiles import StaticFiles

# =========================
# APP INITIALIZATION
# =========================
app = FastAPI(title="Flow Assurance Intelligence Platform (FAIP)")

# Mount static files directory
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================
# FEATURE ENGINEERING FUNCTIONS
# (MATCHING TRAINING PIPELINE EXACTLY)
# ==================================
def _simulate_shutdown_restart(input_dict):
    T_inlet = input_dict['T_inlet']
    P_inlet = input_dict['P_inlet']
    T_seawater = input_dict['T_seawater']
    gas_gravity = input_dict['gas_gravity']
    shutdown_time = input_dict['shutdown_time']

    T_eq = 10 + 15 * np.log(P_inlet) - 25 * gas_gravity
    subcooling = T_eq - T_inlet
    T_final = (T_seawater + (T_inlet - T_seawater) * np.exp(-shutdown_time/8))

    transient = {
        "max_subcooling": subcooling,
        "min_temperature": T_final,
        "time": [0, shutdown_time],
        "temperature_profile": [T_inlet, T_final],
        "hydrate_equilibrium": T_eq
    }
    return transient

def _compute_features(input_dict, transient_data):
    features = input_dict.copy()

    features['T_eq'] = transient_data['hydrate_equilibrium']
    features['subcooling'] = transient_data['max_subcooling']
    features['T_final'] = transient_data['min_temperature']

    features['time_below_eq'] = np.maximum(
        0,
        features['shutdown_time'] * (features['T_eq'] > features['T_final'])
    )

    features['growth_factor'] = features['time_below_eq'] * features['subcooling']

    features['water_fraction'] = features['Q_water'] / (features['Q_water'] + features['Q_oil'] + 1e-6)

    features['inhibition_effect'] = np.exp(-features['chemical_injection']/50)

    features['liquid_flow'] = features['Q_oil'] + features['Q_water']
    
    # IMPORTANT: Convert D_pipe from meters (InputData) to inches (as used in training)
    D_pipe_inches = features['D_pipe'] * 39.3701 # 1 meter = 39.3701 inches
    features['gas_velocity'] = features['Q_gas'] / (D_pipe_inches**2 + 1e-6)

    return features


# =========================
# LOAD MODELS
# =========================
try:
    # Ensure models directory exists
    if not os.path.exists("models"):
        print("Models directory not found. Please ensure models are saved.")
        scaler, classifiers, regressors = None, {}, {}
    else:
        scaler = joblib.load("models/scaler.joblib")
        classifiers = joblib.load("models/classifiers.joblib")
        regressors = joblib.load("models/regressors.joblib")
    print("Models loaded successfully ✅")
except Exception as e:
    print(f"Model loading error: {e}")
    scaler, classifiers, regressors = None, {}, {}


# =========================
# INPUT SCHEMA
# =========================
class InputData(BaseModel):
    T_inlet: float = Field(..., description="Temperature (°C)")
    P_inlet: float = Field(..., description="Pressure (bar)")
    Q_gas: float = Field(..., description="Gas Flow Rate")
    Q_oil: float = Field(..., description="Oil Flow Rate")
    Q_water: float = Field(..., description="Water Flow Rate")
    D_pipe: float = Field(..., description="Pipe Diameter (m)") # Backend expects meters
    T_seawater: float = Field(..., description="Seawater Temperature (°C)")
    gas_gravity: float = Field(..., description="Gas Gravity")
    oil_API: float = Field(..., description="API Gravity")
    salinity: float = Field(..., description="Salinity (%)")
    H2S: float = Field(..., description="H2S fraction")
    CO2: float = Field(..., description="CO2 fraction")
    wax_content: float = Field(..., description="Wax (%)")
    asphaltene_index: float = Field(..., description="Asphaltene Index")
    insulation: int = Field(..., description="0 or 1")
    chemical_injection: float = Field(..., description="Injection Rate")
    age_days: float = Field(..., description="Pipeline Age (days)")
    shutdown_time: float = Field(..., description="Shutdown Time (hours)")


# =========================
# INTELLIGENCE ENGINE
# =========================
def generate_intelligence(result):

    physics = result["physics"]
    transient = result["transient"]
    risk = result["risk_level"]

    max_subcooling = physics.get("max_subcooling", 0)
    min_temp = transient.get("min_temperature", 0)

    insights = []
    observations = []
    recommendations = []

    # INSIGHTS
    if max_subcooling > 20:
        insights.append("Severe hydrate formation risk due to high subcooling.")
    elif max_subcooling > 10:
        insights.append("Moderate hydrate formation risk detected.")
    else:
        insights.append("Low hydrate risk under current conditions.")

    if min_temp < 10:
        insights.append("Pipeline temperature drops significantly during shutdown.")

    # OBSERVATIONS
    observations.append(f"Max subcooling: {round(max_subcooling,2)} °C")
    observations.append(f"Minimum temperature: {round(min_temp,2)} °C")

    # RECOMMENDATIONS
    if risk == "HIGH":
        recommendations.extend([
            "Increase chemical injection immediately",
            "Reduce shutdown duration",
            "Improve insulation",
            "Controlled restart required"
        ])
    elif risk == "MEDIUM":
        recommendations.extend([
            "Monitor hydrate formation",
            "Optimize restart"
        ])
    else:
        recommendations.append("Maintain current operations")

    return {
        "insights": insights,
        "observations": observations,
        "recommendations": recommendations
    }


# =========================
# ROOT
# =========================
@app.get("/")
def home():
    return {"message": "FAIP API running 🚀"}


# =========================
# PREDICTION (FIXED LOGIC)
# =========================
@app.post("/predict")
def predict(data: InputData):

    try:
        if scaler is None or not classifiers or not regressors:
            return JSONResponse(status_code=500, content={"error": "Models not loaded. Please check logs for errors."})

        input_dict = data.dict()

        # ======================
        # TRANSIENT SIMULATION
        # ======================
        transient = _simulate_shutdown_restart(input_dict)

        # ======================
        # ENGINEERING CORRECTION
        # ======================
        max_subcooling = transient.get("max_subcooling", 0)

        # realistic correction
        if input_dict["shutdown_time"] < 8 and input_dict["insulation"] == 1:
            max_subcooling *= 0.6

        transient["max_subcooling"] = max_subcooling

        # ======================
        # FEATURE ENGINEERING
        # ======================
        # This 'features' dictionary now contains all raw and derived features 
        # that the models were trained on.
        features_for_prediction = _compute_features(input_dict, transient)

        # Create DataFrame for scaling and prediction
        df_features = pd.DataFrame([features_for_prediction])
        df_scaled = scaler.transform(df_features)

        # ======================
        # ML PREDICTION
        # ======================
        class_results = {
            k: int(v.predict(df_scaled)[0])
            for k, v in classifiers.items()
        }

        reg_results = {
            k: float(v.predict(df_scaled)[0])
            for k, v in regressors.items()
        }

        # ======================
        # HYBRID RISK MODEL
        # ======================
        ml_score = sum(class_results.values())

        physics_score = (
            2 if max_subcooling > 20 else
            1 if max_subcooling > 10 else
            0
        )

        total_score = ml_score + physics_score

        if total_score >= 3:
            risk_level = "HIGH"
        elif total_score == 2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        hydrate_flag = bool(max_subcooling > 10) # Cast to Python bool

        # ======================
        # RESULT
        # ======================
        result = {
            "classification": class_results,
            "severity": reg_results,
            "risk_level": risk_level,
            "physics": {
                "max_subcooling": max_subcooling,
                "hydrate_risk_flag": hydrate_flag
            },
            "transient": transient
        }

        # ======================
        # ADD INTELLIGENCE
        # ======================
        result.update(generate_intelligence(result))

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "detail": "An unexpected error occurred during prediction."})


# =========================
# REPORT
# =========================
@app.post("/report")
def generate_report(result: dict):

    try:
        os.makedirs("reports", exist_ok=True)

        pdf_path = "reports/FAIP_Report.pdf"
        img_path = "reports/plot.png"

        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(pdf_path)
        content = []

        risk = result["risk_level"]
        transient = result["transient"]

        time = transient.get("time", [])
        temp = transient.get("temperature_profile", [])
        eq = transient.get("hydrate_equilibrium", 0)

        # Plot
        if time and temp and eq:
            plt.figure()
            plt.plot(time, temp, label='Temperature Profile')
            plt.plot(time, [eq]*len(time), linestyle="--", label='Hydrate Equilibrium Temperature')
            plt.xlabel('Time (hours)')
            plt.ylabel('Temperature (°C)')
            plt.title('Temperature Profile during Shutdown')
            plt.legend()
            plt.grid(True)
            plt.savefig(img_path)
            plt.close()
        else:
            print("Warning: Not enough data for transient plot.")

        content.append(Paragraph("FAIP Report", styles['Title']))
        content.append(Spacer(1, 10))
        content.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        content.append(Paragraph(f"Overall Risk Level: <b>{risk}</b>", styles['Normal']))

        content.append(Spacer(1, 10))
        content.append(Paragraph("<b>Insights</b>", styles['Heading2']))
        for i in result.get("insights", []):
            content.append(Paragraph("- " + i, styles['Normal']))
        content.append(Spacer(1, 5))

        content.append(Paragraph("<b>Observations</b>", styles['Heading2']))
        for obs in result.get("observations", []):
            content.append(Paragraph("- " + obs, styles['Normal']))
        content.append(Spacer(1, 5))

        content.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
        for rec in result.get("recommendations", []):
            content.append(Paragraph("- " + rec, styles['Normal']))
        content.append(Spacer(1, 5))

        # Add ML Prediction Details
        content.append(Paragraph("<b>Machine Learning Predictions</b>", styles['Heading2']))
        content.append(Paragraph("<i>Classification (Risk Labels):</i>", styles['Normal']))
        for k, v in result.get('classification', {}).items():
            label_map = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
            content.append(Paragraph(f"- {k.replace('_label', '').capitalize()}: {label_map.get(v, 'UNKNOWN')}", styles['Normal']))
        content.append(Spacer(1, 2))

        content.append(Paragraph("<i>Severity (Probability Scores):</i>", styles['Normal']))
        for k, v in result.get('severity', {}).items():
            content.append(Paragraph(f"- {k.replace('_prob', '').capitalize()}: {v:.2f}", styles['Normal']))
        content.append(Spacer(1, 5))

        # Add plot if generated
        if os.path.exists(img_path):
            content.append(Paragraph("<b>Temperature Profile during Shutdown</b>", styles['Heading2']))
            content.append(Image(img_path, width=450, height=250))

        doc.build(content)

        return FileResponse(pdf_path, media_type="application/pdf", filename="FAIP_Report.pdf")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "detail": "An unexpected error occurred during report generation."})
