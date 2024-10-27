import pandas as pd
from sklearn.decomposition import PCA


def run_pca(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # Initialise PCA and apply with 95% variance explained
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Create new column names for the principal components
    new_column_names = [f'PC_{i+1}' for i in range(X_train_pca.shape[1])]

    # Create DataFrames for the PCA-transformed data
    X_train = pd.DataFrame(X_train_pca, columns=new_column_names)
    X_test = pd.DataFrame(X_test_pca, columns=new_column_names)

    # Print information about the selected components
    final_components = pca.n_components_
    exp_var_ratio = pca.explained_variance_ratio_
    print(
        f'[run_pca] - INFO: Using {final_components} principal components, we can explain {exp_var_ratio.sum() * 100:.2f}% of the variance in the original data'
    )

    return X_train, X_test, pca
