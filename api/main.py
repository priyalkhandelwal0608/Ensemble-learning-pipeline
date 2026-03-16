from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib, numpy as np, pandas as pd

app = FastAPI()
rf, gb, meta = joblib.load("models/ensemble.pkl")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Credit Card Fraud Detection</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%);
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 8px 20px rgba(0,0,0,0.2);
                width: 400px;
                text-align: center;
            }
            h1 {
                color: #333;
                margin-bottom: 20px;
            }
            input[type="text"] {
                width: 90%;
                padding: 10px;
                margin: 8px 0;
                border: 1px solid #ccc;
                border-radius: 6px;
            }
            input[type="submit"] {
                background: #4CAF50;
                color: white;
                padding: 12px 20px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 15px;
            }
            input[type="submit"]:hover {
                background: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Fraud Detection Form</h1>
            <form action="/predict_form" method="post">
                <input type="text" name="Ia" placeholder="Enter Ia"><br>
                <input type="text" name="Ib" placeholder="Enter Ib"><br>
                <input type="text" name="Ic" placeholder="Enter Ic"><br>
                <input type="text" name="Va" placeholder="Enter Va"><br>
                <input type="text" name="Vb" placeholder="Enter Vb"><br>
                <input type="text" name="Vc" placeholder="Enter Vc"><br>
                <input type="submit" value="Predict Fraud">
            </form>
        </div>
    </body>
    </html>
    """

@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(Ia: float = Form(...), Ib: float = Form(...), Ic: float = Form(...),
                 Va: float = Form(...), Vb: float = Form(...), Vc: float = Form(...)):
    df = pd.DataFrame([{"Ia": Ia, "Ib": Ib, "Ic": Ic, "Va": Va, "Vb": Vb, "Vc": Vc}])
    rf_prob = rf.predict_proba(df)[:,1]
    gb_prob = gb.predict_proba(df)[:,1]
    stacked_X = np.column_stack((rf_prob, gb_prob))
    prediction = meta.predict(stacked_X)[0]
    confidence = meta.predict_proba(stacked_X)[0][prediction]

    return f"""
    <html>
    <head><title>Prediction Result</title></head>
    <body style="font-family: Arial; background:#f0f2f5; text-align:center; padding:50px;">
        <h2 style="color:#333;">Prediction Result</h2>
        <p style="font-size:20px;">Prediction: <b>{prediction}</b></p>
        <p style="font-size:20px;">Confidence: <b>{confidence:.2f}</b></p>
        <a href="/" style="display:inline-block; margin-top:20px; padding:10px 20px; background:#4CAF50; color:white; text-decoration:none; border-radius:6px;">Back to Form</a>
    </body>
    </html>
    """