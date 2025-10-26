# ==============================================================
# Dimensionality Reduction with Automatic Target Detection
# ==============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# -------------------------------
# 1. Load dataset
# -------------------------------
file_path = "6customers.csv"  # replace with your file path
data = pd.read_csv(file_path)
print(f"✅ Dataset Loaded | Shape: {data.shape}")

# -------------------------------
# 2. Automatically detect target column (binary categorical)
# -------------------------------
target_col = None
for col in data.columns:
    if data[col].nunique() == 2:
        target_col = col
        break

if target_col:
    print(f"✅ Target column detected: {target_col}")
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Encode target if categorical
    if y.dtype == 'object' or y.nunique() == 2:
        le = LabelEncoder()
        y = le.fit_transform(y)
else:
    print("⚠ No target column detected. Proceeding without target.")
    X = data.copy()
    y = None

# -------------------------------
# 3. Identify categorical and numerical columns
# -------------------------------
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# -------------------------------
# 4. Pipelines for preprocessing
# -------------------------------
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Preprocess features
X_processed = preprocessor.fit_transform(X)
print(f"Shape before dimensionality reduction: {X_processed.shape}")

# -------------------------------
# 5. Dimensionality Reduction
# -------------------------------
n_components = min(50, X_processed.shape[1])
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd.fit_transform(X_processed)
print(f"Shape after dimensionality reduction: {X_reduced.shape}")

# Optional: Explained variance
explained_variance = svd.explained_variance_ratio_.sum()
print(f"Total Explained Variance: {explained_variance:.4f}")

# -------------------------------
# 6. Classification (if target exists)
# -------------------------------
if y is not None:
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n--- PERFORMANCE METRICS ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Convert target names to strings to avoid TypeError
    if 'le' in locals():  # target was label-encoded
        target_names = [str(cls) for cls in le.classes_]
    else:  # numeric target
        target_names = [str(cls) for cls in np.unique(y)]

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=target_names))
else:
    print("No target column available. Dimensionality reduction completed successfully.")
