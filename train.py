import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)

# ── 1. Rebuild preprocessed data ─────────────────────────
df = pd.read_csv("student-mat.csv", sep=";")
df["G3"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

X = df.drop(columns=["G3"])
y = df["G3"]

binary_cols = ["school", "sex", "address", "famsize",
               "Pstatus", "schoolsup", "famsup", "paid",
               "activities", "nursery", "higher", "internet", "romantic"]
le = LabelEncoder()
for col in binary_cols:
    X[col] = le.fit_transform(X[col])

multi_cols = ["Mjob", "Fjob", "reason", "guardian"]
X = pd.get_dummies(X, columns=multi_cols)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. Define models ──────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "KNN (K=5)":           KNeighborsClassifier(n_neighbors=5)
}

# ── 3. Train, evaluate, store results ────────────────────
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc        = accuracy_score(y_test, y_pred)
    prec_pass  = precision_score(y_test, y_pred, pos_label=1)
    prec_fail  = precision_score(y_test, y_pred, pos_label=0)
    rec_pass   = recall_score(y_test, y_pred, pos_label=1)
    rec_fail   = recall_score(y_test, y_pred, pos_label=0)
    f1_pass    = f1_score(y_test, y_pred, pos_label=1)
    f1_fail    = f1_score(y_test, y_pred, pos_label=0)
    macro_f1   = f1_score(y_test, y_pred, average="macro")
    cm         = confusion_matrix(y_test, y_pred)

    results[name] = {
        "model":      model,
        "macro_f1":   macro_f1,
        "metrics": {
            "Accuracy":         acc,
            "Precision (Pass)": prec_pass,
            "Precision (Fail)": prec_fail,
            "Recall (Pass)":    rec_pass,
            "Recall (Fail)":    rec_fail,
            "F1 Score (Pass)":  f1_pass,
            "F1 Score (Fail)":  f1_fail,
            "Macro Avg F1":     macro_f1,
        },
        "confusion_matrix": cm,
        "report": classification_report(y_test, y_pred,
                                        target_names=["Fail", "Pass"])
    }

# ── 4. Print results ──────────────────────────────────────
print("=" * 60)
print("COMPARATIVE PERFORMANCE MATRIX")
print("=" * 60)

metric_names = list(list(results.values())[0]["metrics"].keys())
col_w = 22

header = f"{'METRIC':<25}" + "".join(f"{n:>{col_w}}" for n in results.keys())
print(header)
print("-" * len(header))

for metric in metric_names:
    row = f"{metric:<25}"
    for name in results:
        val = results[name]["metrics"][metric]
        row += f"{val:>{col_w}.4f}"
    print(row)

print("\n" + "=" * 60)
print("CONFUSION MATRICES")
print("=" * 60)
for name, data in results.items():
    cm = data["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{name}")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Matrix:\n{cm}")

print("\n" + "=" * 60)
print("FULL CLASSIFICATION REPORTS")
print("=" * 60)
for name, data in results.items():
    print(f"\n── {name} ──")
    print(data["report"])

# ── 5. Select best model by macro F1 ─────────────────────
best_name = max(results, key=lambda n: results[n]["macro_f1"])
best_model = results[best_name]["model"]
best_f1    = results[best_name]["macro_f1"]

print("=" * 60)
print(f"BEST MODEL: {best_name}")
print(f"MACRO F1:   {best_f1:.4f}")
print("=" * 60)

# ── 6. Serialize best model and scaler ───────────────────
joblib.dump(best_model, "best_model.joblib")
joblib.dump(scaler,     "scaler.joblib")
joblib.dump(list(X.columns), "feature_columns.joblib")

print(f"\n{best_name} saved as best_model.joblib")
print("Scaler saved as scaler.joblib")
print("Feature columns saved as feature_columns.joblib")

