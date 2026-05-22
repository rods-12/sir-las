import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

model   = joblib.load("best_model.joblib")
scaler  = joblib.load("scaler.joblib")
columns = joblib.load("feature_columns.joblib")

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

def generate_suggestions(form_data):
    suggestions = []

    if form_data["G2"] < 10:
        suggestions.append({
            "field": "Second Period Grade (G2)",
            "current": form_data["G2"],
            "target": "10 or above",
            "advice": "Your G2 grade is below the passing threshold. Focus on reviewing your G2 subject material, seek help from your teacher, and consider joining a study group."
        })

    if form_data["G1"] < 10:
        suggestions.append({
            "field": "First Period Grade (G1)",
            "current": form_data["G1"],
            "target": "10 or above",
            "advice": "Your G1 grade indicates early academic struggles. Talk to your teacher about what topics need the most attention and address those gaps now before they compound."
        })

    if form_data["studytime"] < 2:
        suggestions.append({
            "field": "Study Time",
            "current": f"Level {form_data['studytime']} (low)",
            "target": "Level 2 or higher",
            "advice": "You are spending very little time studying. Try setting aside at least 2 hours of focused study time per day. Avoid distractions like social media during study sessions."
        })

    if form_data["failures"] > 0:
        suggestions.append({
            "field": "Past Failures",
            "current": f"{form_data['failures']} failure(s)",
            "target": "0 failures",
            "advice": "Past failures significantly impact your prediction. Meet with your academic advisor or teacher to create a recovery plan and identify which subjects need the most attention."
        })

    if form_data["absences"] > 10:
        suggestions.append({
            "field": "Absences",
            "current": f"{form_data['absences']} absences",
            "target": "Below 10",
            "advice": "High absences mean you are missing critical lessons. Prioritize attendance. If absences are due to personal issues, speak with your school counselor for support."
        })

    if form_data["higher"] == 0:
        suggestions.append({
            "field": "Desire for Higher Education",
            "current": "No",
            "target": "Yes",
            "advice": "Students who want to pursue higher education tend to perform better because they have a clear goal. Consider setting long-term academic and career goals to motivate yourself."
        })

    if form_data["internet"] == 0:
        suggestions.append({
            "field": "Internet Access",
            "current": "No",
            "target": "Yes",
            "advice": "Lack of internet access limits your ability to do research and access learning materials. Ask your school if they provide internet access or visit your local library."
        })

    if form_data["health"] < 3:
        suggestions.append({
            "field": "Health Status",
            "current": f"Level {form_data['health']} (poor)",
            "target": "Level 3 or above",
            "advice": "Poor health directly affects your ability to study and concentrate. Make sure you are getting enough sleep, eating properly, and seeking medical help if needed."
        })

    # If somehow no specific suggestion triggered
    if not suggestions:
        suggestions.append({
            "field": "General",
            "current": "Multiple weak areas",
            "target": "Overall improvement",
            "advice": "Focus on improving your grades, attending all classes, and increasing your study time. Speak with your teacher for personalized guidance."
        })

    return suggestions


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
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

        row = BASELINE.copy()
        row.update(form_data)

        input_df     = pd.DataFrame([row])[columns]
        input_scaled = scaler.transform(input_df.values)

        prediction  = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        result     = "PASS" if prediction == 1 else "FAIL"
        confidence = round(probability[prediction] * 100, 2)

        suggestions = generate_suggestions(form_data) if result == "FAIL" else []

        return render_template("result.html",
                               result=result,
                               confidence=confidence,
                               form_data=form_data,
                               suggestions=suggestions)
    except Exception as e:
        return f"Error: {str(e)}", 400


if __name__ == "__main__":
    app.run(debug=True)