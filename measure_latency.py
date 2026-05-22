import joblib
import pandas as pd
import time

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

# 3 different test cases
test_cases = [
    {"studytime": 4, "failures": 0, "absences": 2,  "higher": 1, "internet": 1, "G1": 15, "G2": 16, "health": 5},
    {"studytime": 1, "failures": 3, "absences": 20, "higher": 0, "internet": 0, "G1": 5,  "G2": 4,  "health": 2},
    {"studytime": 2, "failures": 1, "absences": 8,  "higher": 1, "internet": 1, "G1": 10, "G2": 11, "health": 3},
]

print("=" * 55)
print("INFERENCE LATENCY TEST")
print("=" * 55)

latencies = []

for i, case in enumerate(test_cases, 1):
    row = BASELINE.copy()
    row.update(case)
    input_df = pd.DataFrame([row])[columns]

    start = time.perf_counter()
    # input_scaled = scaler.transform(input_df)
    input_scaled = scaler.transform(input_df.values) 
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]
    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)

    result     = "PASS" if prediction == 1 else "FAIL"
    confidence = round(probability[prediction] * 100, 2)

    print(f"\nTest Case {i}:")
    print(f"  Input:      {case}")
    print(f"  Result:     {result} ({confidence}% confidence)")
    print(f"  Latency:    {latency_ms:.4f} ms")

print("\n" + "=" * 55)
print(f"Average Latency: {sum(latencies)/len(latencies):.4f} ms")
print(f"Max Latency:     {max(latencies):.4f} ms")
print(f"Min Latency:     {min(latencies):.4f} ms")
print("All well under 3000ms (3 second) threshold." if max(latencies) < 3000 else "WARNING: Exceeded 3 second threshold.")
print("=" * 55)
