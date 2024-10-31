# DSA4262 Genomics Project: Prediction of m6A RNA modifications from direct RNA-Seq data

This repository contains the code for the DSA4262 Genomics Project by Team Gene-ius. The pipeline supports data preparation, feature engineering, model training, and evaluation, which can be run as standalone modules or sequentially through curated scripts. More details are provided below.

## Machine Prerequisites

-   **AWS Instance**: TO BE CONFIRMED
-   **Local Machine**: TO BE CONFIRMED

## Set Up Repository

1. Clone the repository

```bash
git clone https://github.com/AY2425S1-DSA4262-gene-ius/Genomics-Project.git
```

2. Navigate to the project directory

```bash
cd Genomics-Project
```

## Package Installation

### Installing `pip`

`pip` is a package installer for python. We will utilise it for any package dependencies in the project.

-   **AWS Instance**:

```
TO BE CONFIRMED
```

-   **Local Machine**: If Python is already installed, `pip` should be included automatically.

### **OPTIONAL** - Setup Local Environment

We strongly encourage you to utilise a local environment for package installation before proceeding with the steps.

Refer to the following links for possible setup of the local environment:

-   [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
-   [venv](https://docs.python.org/3/library/venv.html)

If you have set up your enviroment, or are not concerned with package isolation, you may continue with the follow steps.

### Installing packages

While in `Genomics-Project` folder, run:

```bash
pip install -r requirements.txt
```

You’re now set to run training and make predictions using our pipeline.

## Model Training Pipeline

### Data Prerequisites

Before running the full training process, ensure that you have both the `.json.gz` file and the associated `.labelled` file that you intend to use for training of the model.

### Run Training Pipeline

The main pipeline logic, which sequentially handles data preparation, feature engineering, model training, and evaluation, is located in `model_training.py`.

Suppose you have `dataset0.json.gz` and `data.info.labelled` in the `data` folder, you can quickly initalise the pipeline with the command below:

```bash
python -m model_training --data_file_path data/dataset0.json.gz --labels_data_path data/data.info.labelled --output_file_name training_output.csv
```

For further modifications to the parameters, you may refer to the flags below:
| Flag | Type | Description | Default |
|---------------------|--------|---------------------------------------------------|---------|
| `--data_file_path` | `str` | Path to the gzipped dataset JSON file. | None |
| `--labels_data_path`| `str` | Path to the labels file. | None |
| `--output_file_name`| `str` | Filename for the output. | None |
| `--train_data_ratio`| `float`| Ratio for train data in train-test split. | `0.8` |
| `--threshold` | `float`| Probabilistic threshold for binary classification.| `0.5` |
| `--seed` | `int` | Seed for reproducibility. | `42` |

If you wish, you may run the following command for information about the flags:

```bash
python -m model_training --help
```

## Prediction Pipeline

> [!IMPORTANT]
> If you are here to test the prediction process, the trained model and artifacts are already in the repository. You may copy and directly run the command below for prediction on our sample data:
>
> ```bash
> python -m make_predictions --data_file_path data/sample_data.json.gz --model_path models/Histogram-based_Gradient_Boosting.joblib --standard_scaler_path artifacts/standard_scaler.joblib --pca_path artifacts/pca.joblib --output_file_name sample_data_predictions.csv
> ```
>
> After running the command, you should expect to see a generated output file in `predictions/sample_data_predictions.csv` that contains all the predictions. Feel free to follow the steps below if you wish to use your own dataset.

### Data Prerequisites

Before running predictions, there are some files which are necessary. Running the Model Training Pipeline will also generate these files:

-   `{DATA_FILENAME}.json.gz` file for the reads data
-   `model.joblib` file that holds the trained model
-   `standard_scaler.joblib` file that holds the fitted scaler for feature standardisation
-   `pca.joblib` file that holds the fitted PCA artifact for Principal Component Analysis on the features

Our sample dataset, model and its artifacts are located at:

-   `data/sample_data.json.gz`
-   `models/Histogram-based_Gradient_Boosting.joblib`
-   `artifacts/standard_scaler.joblib`
-   `artifacts/pca.joblib`

### Run Evaluation Pipeline

With the necessary files, you may modify the paths in the command below and run it:

```bash
python -m make_predictions --data_file_path data/sample_data.json.gz --model_path models/Histogram-based_Gradient_Boosting.joblib --standard_scaler_path artifacts/standard_scaler.joblib --pca_path artifacts/pca.joblib --output_file_name sample_data_predictions.csv
```

The predictions will then be generated in the `predictions` folder with your indicated output file name.

The flags for the command that executes the prediction pipeline are as follows:
| Flag | Type | Description | Default |
| ------------------------ | ----- | -------------------------------------- | ------- |
| `--data_file_path` | `str` | Path to the gzipped dataset JSON file. | None |
| `--model_path` | `str` | Path to the trained model. | None |
| `--standard_scaler_path` | `str` | Path to the fitted StandardScaler. | None |
| `--pca_path` | `str` | Path to the fitted PCA artifact. | None |
| `--output_file_name` | `str` | Filename for the output. | None |

### Running Individual Components

If you are interested in executing the individual components of **data preparation**, **feature engineering**, **model training**, and **evaluation**, you may refer to the sample commands below. Else, head to `m6a_modifications` folder and refer to the the various component scripts for more information.

Note that the commands are run in the root directory: `Genomics-Project`.

```bash
# Data Preparation
python -m m6a_modifications.raw_data_preparer --data_file_path data/dataset0.json.gz

# Feature Engineering
python -m m6a_modifications.data_processing --reads_data_path data/dataset0.json.gz.csv --labels_data_path data/data.info.labelled

# Model Training
python -m m6a_modifications.modelling --x_train_data_path data/X_train.csv --y_train_data_path data/y_train.csv

# Evaluation
python -m m6a_modifications.evaluation --model_file_path models/Histogram-based_Gradient_Boosting.joblib --x_test_data_path data/X_test.csv --y_test_data_path data/y_test.csv
```
