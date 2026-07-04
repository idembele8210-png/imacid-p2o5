# =====================================================================
# IMACID — API REST FastAPI
# Endpoints : /predict  /predict/batch  /health  /model/info
# Deploiement : Render.com (gratuit)
# =====================================================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import joblib
import os
from pathlib import Path
from datetime import datetime

# ── Chargement modele ─────────────────────────────────────────────
MODEL_PATH = Path(os.getenv("MODEL_PATH",
               "IMACID_Resultats/modeles/XGBoost_champion.joblib"))

if not MODEL_PATH.exists():
    raise FileNotFoundError("Modele introuvable : " + str(MODEL_PATH))

model = joblib.load(MODEL_PATH)
RMSE  = 0.1978
LOAD_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ── Constantes ────────────────────────────────────────────────────
ALL_FEATURES = [
    "Densite","TS","CO2","SiO2_T","SiO2_R","CaO","SO4","F_",
    "Fe2O3","Al2O3","K2O","MgO","Na2O","Cd",
    "G500","G400","G315","G250","G160","G125","G80","G40"
]
DEFAULTS = {
    "Densite":1.65,"TS":65.0,"CO2":5.2,"SiO2_T":14.5,"SiO2_R":13.0,
    "CaO":48.0,"SO4":2.8,"F_":3.1,"Fe2O3":2.9,"Al2O3":4.0,
    "K2O":0.78,"MgO":1.45,"Na2O":0.65,"Cd":0.001,
    "G500":2.0,"G400":5.0,"G315":12.0,"G250":25.0,
    "G160":55.0,"G125":70.0,"G80":85.0,"G40":98.0,
}

# ── App FastAPI ───────────────────────────────────────────────────
app = FastAPI(
    title="IMACID P2O5 Prediction API",
    description="API de prediction de la teneur en P2O5 — Modele XGBoost",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas Pydantic ──────────────────────────────────────────────
class SampleInput(BaseModel):
    Densite: float = Field(default=1.65, ge=1.2, le=2.2)
    TS:      float = Field(default=65.0, ge=40,  le=85)
    CO2:     float = Field(default=5.2,  ge=0,   le=15)
    SiO2_T:  float = Field(default=14.5, ge=5,   le=25)
    SiO2_R:  float = Field(default=13.0, ge=4,   le=22)
    CaO:     float = Field(default=48.0, ge=35,  le=58)
    SO4:     float = Field(default=2.8,  ge=0.5, le=6)
    F_:      float = Field(default=3.1,  ge=1,   le=6)
    Fe2O3:   float = Field(default=2.9,  ge=1.5, le=5)
    Al2O3:   float = Field(default=4.0,  ge=1.5, le=7)
    K2O:     float = Field(default=0.78, ge=0.4, le=1.2)
    MgO:     float = Field(default=1.45, ge=0.8, le=3.5)
    Na2O:    float = Field(default=0.65, ge=0.2, le=1.5)
    Cd:      float = Field(default=0.001,ge=0,   le=0.01)
    G500:    float = Field(default=2.0,  ge=0,   le=20)
    G400:    float = Field(default=5.0,  ge=0,   le=30)
    G315:    float = Field(default=12.0, ge=0,   le=40)
    G250:    float = Field(default=25.0, ge=0,   le=60)
    G160:    float = Field(default=55.0, ge=10,  le=90)
    G125:    float = Field(default=70.0, ge=30,  le=95)
    G80:     float = Field(default=85.0, ge=50,  le=99)
    G40:     float = Field(default=98.0, ge=80,  le=100)

class PredictionResult(BaseModel):
    p2o5_predit:  float
    qualite:      str
    ic_low_95:    float
    ic_high_95:   float
    timestamp:    str
    interpretation: str

class BatchInput(BaseModel):
    samples: List[SampleInput]

class BatchResult(BaseModel):
    n_samples:    int
    p2o5_moyen:   float
    p2o5_min:     float
    p2o5_max:     float
    predictions:  List[PredictionResult]
    distribution: dict

# ── Fonction de prediction ────────────────────────────────────────
def _predict_one(sample: SampleInput) -> PredictionResult:
    X = np.array([[getattr(sample, f) for f in ALL_FEATURES]])
    p = float(model.predict(X)[0])
    if   p >= 31.0: q = "PREMIUM";  interp = "Qualite excellente — export premium"
    elif p >= 29.5: q = "STANDARD"; interp = "Qualite standard — conforme export"
    elif p >= 28.0: q = "LIMITE";   interp = "Qualite limite — verifier avant export"
    else:           q = "ALERTE";   interp = "Qualite insuffisante — lot a retenir"
    return PredictionResult(
        p2o5_predit  = round(p, 4),
        qualite      = q,
        ic_low_95    = round(p - 1.96*RMSE, 4),
        ic_high_95   = round(p + 1.96*RMSE, 4),
        timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        interpretation = interp
    )

# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html><head><title>IMACID API</title></head>
    <body style="background:#0D1117;color:#F0F6FC;font-family:monospace;padding:40px;">
    <h1 style="color:#F0B429;">IMACID — P2O5 Prediction API</h1>
    <p>Modele XGBoost | R2=0.8694 | RMSE=0.1978%</p>
    <ul>
      <li><a href="/docs"  style="color:#3498DB;">/docs  — Documentation interactive (Swagger)</a></li>
      <li><a href="/redoc" style="color:#3498DB;">/redoc — Documentation alternative</a></li>
      <li><a href="/health"style="color:#2ECC71;">/health — Statut API</a></li>
    </ul>
    </body></html>
    """

@app.get("/health")
def health():
    return {
        "status":     "OK",
        "model":      "XGBoost_champion",
        "r2":         0.8694,
        "rmse":       0.1978,
        "loaded_at":  LOAD_TIME,
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/model/info")
def model_info():
    return {
        "nom":         "XGBoost Regressor",
        "configuration": "Nettoyé",
        "r2_test":     0.8694,
        "rmse":        0.1978,
        "mae":         0.1639,
        "rpd":         2.77,
        "biais":       0.0153,
        "n_features":  22,
        "n_samples":   366,
        "features":    ALL_FEATURES,
        "seuils": {
            "PREMIUM":  "p2o5 >= 31.0",
            "STANDARD": "29.5 <= p2o5 < 31.0",
            "LIMITE":   "28.0 <= p2o5 < 29.5",
            "ALERTE":   "p2o5 < 28.0"
        }
    }

@app.post("/predict", response_model=PredictionResult)
def predict_single(sample: SampleInput):
    try:
        return _predict_one(sample)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchResult)
def predict_batch(batch: BatchInput):
    if not batch.samples:
        raise HTTPException(status_code=400, detail="Liste vide")
    if len(batch.samples) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 echantillons")
    try:
        results = [_predict_one(s) for s in batch.samples]
        preds   = [r.p2o5_predit for r in results]
        quals   = [r.qualite for r in results]
        dist    = {}
        for q in ["PREMIUM","STANDARD","LIMITE","ALERTE"]:
            cnt = quals.count(q)
            dist[q] = {"count": cnt, "pct": round(cnt/len(quals)*100, 1)}
        return BatchResult(
            n_samples   = len(results),
            p2o5_moyen  = round(float(np.mean(preds)), 4),
            p2o5_min    = round(float(np.min(preds)), 4),
            p2o5_max    = round(float(np.max(preds)), 4),
            predictions = results,
            distribution= dist
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api_fastapi:app", host="0.0.0.0", port=port, reload=False)
