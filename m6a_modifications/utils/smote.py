import pandas as pd
from imblearn.over_sampling import SMOTE


def run_smote(X_train: pd.DataFrame, y_train: pd.DataFrame, seed: int = 42):
    # Initialise SMOTE and oversample
    smote = SMOTE(random_state=seed)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, y_train
