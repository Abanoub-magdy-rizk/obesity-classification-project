import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CSV_PATH = r"..\..\data\ObesityDataSet_raw_and_data_sinthetic.csv"
df = pd.read_csv(CSV_PATH)

TARGET_COL = "NObeyesdad"
X = df.drop(columns=[TARGET_COL]).copy()
y = df[TARGET_COL].astype(str).copy()

if "Weight" in X.columns and "Height" in X.columns:
    med_h = X["Height"].median()
    if med_h > 3:
        height_m = X["Height"] / 100.0
    else:
        height_m = X["Height"]
    height_m = height_m.replace(0, np.nan)
    X["BMI"] = X["Weight"] / (height_m ** 2)

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

try:
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
except:
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder="drop"
)

X_proc = preprocessor.fit_transform(X)

if hasattr(X_proc, "toarray"):
    X_proc = X_proc.toarray()

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_proc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)