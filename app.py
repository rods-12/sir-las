import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model, scaler, and feature columns
model    = joblib.load("best_model.joblib")
scaler   = joblib.load("scaler.joblib")
columns  = joblib.load("feature_columns.joblib")

# Baseline defaults for all 45 columns
BASELINE = {
    "school": 0, "sex": 0, "age": 17, "address": 1, "famsize": 1,
    "Pstatus": 1, "Medu": 2, "Fedu": 2, "traveltime": 2, "studytime": 2,
    "failures": 0, "schoolsup": 0, "famsup": 1, "paid": 0,
    "activities": 0, "nursery": 1, "higher": 1, "internet": 1,
    "romantic": 0, "famrel": 4, "freetime": 3, "goout": 3,
    "Dalc": 1, "Walc": 1, "health": 3, "absences": 0,
    "G1": 10, "G2": 10,
    "Mjob_at_home": 0, "Mjob_health": 0, "Mjob_other": 1,
    "Mjob_services": 0, "Mjob_teacher": 0,
    "Fjob_at_home": 0, "Fjob_health": 0, "Fjob_other": 1,
    "Fjob_services": 0, "Fjob_teacher": 0,
    "reason_course": 1, "reason_home": 0, "reason_other": 0,
    "reason_reputation": 0,
    "guardian_father": 0, "guardian_mother": 1, "guardian_other": 0
}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect 8 inputs from form
        form_data = {
            "studytime": int(request.form["studytime"]),
            "failures":  int(request.form["failures"]),
            "absences":  int(request.form["absences"]),
            "higher":    int(request.form["higher"]),
            "internet":  int(request.form["internet"]),
            "G1":        int(request.form["G1"]),
            "G2":        int(request.form["G2"]),
            "health":    int(request.form["health"]),
        }

        # Build full feature row from baseline
        row = BASELINE.copy()
        row.update(form_data)

        # Convert to DataFrame with correct column order
        input_df = pd.DataFrame([row])[columns]

        # Scale
        # input_scaled = scaler.transform(input_df)
        input_scaled = scaler.transform(input_df.values)

        # Predict
        prediction  = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        result = "WILL PASS" if prediction == 1 else "WILL FAIL"
        confidence = round(probability[prediction] * 100, 2)

        return render_template("result.html",
                               result=result,
                               confidence=confidence,
                               form_data=form_data)
    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == "__main__":
    app.run(debug=True)