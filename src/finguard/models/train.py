from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

from finguard.config import MODEL


def build_model_pipeline():
    """Build the preprocessing ColumnTransformer and XGBClassifier pipeline.

    Reads feature lists and XGBoost settings from config and returns an unfitted Pipeline.
    """
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, MODEL["numerical_features"]),
            ("cat", categorical_transformer, MODEL["categorical_features"]),
        ]
    )

    classifier = xgb.XGBClassifier(**MODEL["xgb_params"])
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    return pipeline


def train_model(pipeline, X_train, y_train):
    """Fit the preprocessing and XGBoost pipeline on the training data.

    Takes an unfitted pipeline, X_train and y_train, and returns the fitted pipeline.
    """
    pipeline.fit(X_train, y_train)
    return pipeline
