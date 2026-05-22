import joblib
import pandas as pd

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

# Documented test cases with expected outcomes
test_cases = [
    {
        "label": "High Performer",
        "expected": "PASS",
        "inputs": {"studytime": 4, "failures": 0, "absences": 1,
                   "higher": 1, "internet": 1, "G1": 17, "G2": 18, "health": 5}
    },
    {
        "label": "At Risk Student",
        "expected": "FAIL",
        "inputs": {"studytime": 1, "failures": 3, "absences": 25,
                   "higher": 0, "internet": 0, "G1": 4, "G2": 3, "health": 1}
    },
    {
        "label": "Average Student",
        "expected": "PASS",
        "inputs": {"studytime": 2, "failures": 0, "absences": 6,
                   "higher": 1, "internet": 1, "G1": 11, "G2": 12, "health": 3}
    },
    {
        "label": "Borderline Student",
        "expected": "UNCERTAIN",
        "inputs": {"studytime": 2, "failures": 1, "absences": 10,
                   "higher": 1, "internet": 0, "G1": 9, "G2": 9, "health": 3}
    },
    {
        "label": "Good Grades Bad Habits",
        "expected": "PASS",
        "inputs": {"studytime": 1, "failures": 0, "absences": 15,
                   "higher": 1, "internet": 1, "G1": 14, "G2": 15, "health": 2}
    },
]

print("=" * 65)
print("FORMAL SYSTEM PREDICTION TEST")
print("=" * 65)

passed = 0
for case in test_cases:
    row = BASELINE.copy()
    row.update(case["inputs"])
    input_df     = pd.DataFrame([row])[columns]
    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    result     = "PASS" if prediction == 1 else "FAIL"
    confidence = round(probability[prediction] * 100, 2)
    match      = "✓" if result == case["expected"] else "✗" if case["expected"] != "UNCERTAIN" else "-"
    if match == "✓":
        passed += 1

    print(f"\n{case['label']}")
    print(f"  Inputs:     {case['inputs']}")
    print(f"  Expected:   {case['expected']}")
    print(f"  Predicted:  {result} ({confidence}% confidence)")
    print(f"  Match:      {match}")

print("\n" + "=" * 65)
print(f"Results: {passed}/4 expected outcomes matched (excluding UNCERTAIN)")
print("=" * 65)
