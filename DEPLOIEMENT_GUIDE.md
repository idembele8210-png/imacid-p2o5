# IMACID — Guide de Deploiement Complet
## Interface Streamlit + API FastAPI — Internet Public

---

## ETAPE 1 — Preparer les fichiers localement

Structure requise sur ton PC :
```
imacid-deploiement/
├── app_streamlit.py
├── api_fastapi.py
├── requirements.txt
└── IMACID_Resultats/
    └── modeles/
        └── XGBoost_champion.joblib
```

---

## ETAPE 2 — Creer le depot GitHub

1. Aller sur https://github.com → New repository
2. Nom : `imacid-p2o5`
3. Visibilite : Public (requis pour Streamlit Cloud gratuit)
4. Creer le depot

Puis dans ton terminal :
```bash
cd C:/Users/LENOVO/Documents/imacid-deploiement
git init
git add .
git commit -m "IMACID deploiement initial"
git branch -M main
git remote add origin https://github.com/TON_USERNAME/imacid-p2o5.git
git push -u origin main
```

---

## ETAPE 3 — Deployer l interface sur Streamlit Cloud

URL : https://streamlit.io

1. Se connecter avec ton compte GitHub
2. Cliquer "New app"
3. Selectionner : Repository = `imacid-p2o5`
4. Branch = `main`
5. Main file path = `app_streamlit.py`
6. Cliquer "Deploy"

Resultat : `https://TON_USERNAME-imacid-p2o5-app-streamlit.streamlit.app`

Delai de deploiement : 2-4 minutes

---

## ETAPE 4 — Deployer l API sur Render.com

URL : https://render.com

1. Se connecter avec GitHub
2. "New" → "Web Service"
3. Connecter le depot `imacid-p2o5`
4. Configurer :
   - Name        : `imacid-api`
   - Runtime      : Python 3
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn api_fastapi:app --host 0.0.0.0 --port $PORT`
5. Plan : Free
6. Cliquer "Create Web Service"

Resultat : `https://imacid-api.onrender.com`
Documentation : `https://imacid-api.onrender.com/docs`

Delai de deploiement : 3-5 minutes

---

## ETAPE 5 — Tester l API depuis Excel (VBA)

```vba
Sub PredireP2O5()
    Dim http As Object
    Set http = CreateObject("MSXML2.XMLHTTP")

    Dim url As String
    url = "https://imacid-api.onrender.com/predict"

    Dim body As String
    body = "{"
    body = body & Chr(34) & "Densite" & Chr(34) & ":" & Cells(2,1).Value & ","
    body = body & Chr(34) & "TS"      & Chr(34) & ":" & Cells(2,2).Value & ","
    body = body & Chr(34) & "CaO"     & Chr(34) & ":" & Cells(2,3).Value
    body = body & "}"

    http.Open "POST", url, False
    http.setRequestHeader "Content-Type", "application/json"
    http.Send body

    Dim result As String
    result = http.responseText

    MsgBox "Resultat : " & result
End Sub
```

---

## ETAPE 6 — Tester l API avec cURL

```bash
# Test sante
curl https://imacid-api.onrender.com/health

# Prediction unique
curl -X POST https://imacid-api.onrender.com/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"Densite\":1.68,\"TS\":67.5,\"CaO\":49.5}"

# Prediction batch (plusieurs echantillons)
curl -X POST https://imacid-api.onrender.com/predict/batch ^
  -H "Content-Type: application/json" ^
  -d "{\"samples\":[{\"Densite\":1.68,\"TS\":67.5},{\"Densite\":1.72,\"TS\":66.0}]}"
```

---

## RESUME DES URLs apres deploiement

| Service | URL | Usage |
|---------|-----|-------|
| Interface operateurs | https://TON_USERNAME-imacid-p2o5.streamlit.app | Navigateur web |
| API REST | https://imacid-api.onrender.com | Excel / LIMS / Python |
| Documentation API | https://imacid-api.onrender.com/docs | Swagger UI |
| Sante API | https://imacid-api.onrender.com/health | Monitoring |

---

## MISE A JOUR du modele

Quand tu as un nouveau modele :
```bash
# Remplacer le .joblib localement
copy nouveau_modele.joblib IMACID_Resultats/modeles/XGBoost_champion.joblib

# Pousser sur GitHub -> deploiement automatique
git add .
git commit -m "Mise a jour modele XGBoost"
git push
```
Streamlit Cloud et Render se mettent a jour automatiquement en 2-3 minutes.
