
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

data = pd.read_csv("indiancrop_dataset.csv")

print("Dataset Shape:", data.shape)
print(data.head())
print(data.columns.tolist())
data.info()

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:", data.duplicated().sum())

X = data.drop(columns=["CROP", "CROP_PRICE"])
y = data["CROP"]

print("\nInput Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

numeric_features = [
    "N_SOIL",
    "P_SOIL",
    "K_SOIL",
    "TEMPERATURE",
    "HUMIDITY",
    "ph",
    "RAINFALL"
]

categorical_features = ["STATE"]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing Completed Successfully!")
print("Processed Training Shape:", X_train_processed.shape)
print("Processed Testing Shape:", X_test_processed.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)
