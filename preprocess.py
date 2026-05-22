import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ── 1. Load ──────────────────────────────────────────────
df = pd.read_csv("student-mat.csv", sep=";")

print("=== SHAPE ===")
print(df.shape)

print("\n=== NULL CHECK ===")
print(df.isnull().sum())

print("\n=== G3 DISTRIBUTION (before binarize) ===")
print(df["G3"].describe())

# ── 2. Binarize target ───────────────────────────────────
df["G3"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

print("\n=== CLASS DISTRIBUTION ===")
print(df["G3"].value_counts())
print(df["G3"].value_counts(normalize=True).round(3))

# ── 3. Separate features and target ──────────────────────
X = df.drop(columns=["G3"])
y = df["G3"]

# ── 4. Label encode binary categoricals ──────────────────
binary_cols = ["school", "sex", "address", "famsize",
               "Pstatus", "schoolsup", "famsup", "paid",
               "activities", "nursery", "higher", "internet", "romantic"]

le = LabelEncoder()
for col in binary_cols:
    X[col] = le.fit_transform(X[col])

# ── 5. One-hot encode multi-category categoricals ────────
multi_cols = ["Mjob", "Fjob", "reason", "guardian"]
X = pd.get_dummies(X, columns=multi_cols)

print("\n=== FEATURE MATRIX SHAPE AFTER ENCODING ===")
print(X.shape)

# ── 6. Min-Max scale all columns ─────────────────────────
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# ── 7. Stratified 80/20 split ────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\n=== SPLIT SIZES ===")
print(f"Training set: {X_train.shape[0]} records")
print(f"Testing set:  {X_test.shape[0]} records")

print("\n=== TRAINING CLASS DISTRIBUTION ===")
print(y_train.value_counts())

print("\n=== TESTING CLASS DISTRIBUTION ===")
print(y_test.value_counts())

print("\n=== COLUMN LIST ===")
print(list(X_scaled.columns))

print("\nPreprocessing complete.")