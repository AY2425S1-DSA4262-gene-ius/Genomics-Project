# DSA4262 Genomics Project: Prediction of m6A RNA modifications from direct RNA-Seq data

This repository contains the code for the DSA4262 Genomics Project by Team Gene-ius. The pipeline supports data preparation, feature engineering, model training, and evaluation, which can be run as standalone modules or sequentially through curated scripts. More details are provided below.

> [!TIP]
>
> **👋 Hello Student Evaluators!**
>
> Welcome! We understand our README may be lengthy. If you’re short on time (or patience), you can simply run the commands below in sequence to execute the prediction script and generate the output file directly.
>
> </p>
>
> If you're using the AWS instance, just make sure your `EBSVolumeSize` is at least `100` and `InstanceType` is at least `t3.medium`. Also, do SSH in your terminal (`ssh -i /path/to/your/.pem ubuntu@XXX.XXX.XXX`), since the terminal in Research Gateway may be buggy for some.
>
> If you're running locally, make sure you have **Python version 3.9 or later**.
>
> The output file will then be generated in `predictions/sample_data_predictions.csv`

## Summary of Commands (to get output)

### If you're using AWS Instance, set up Python 3.9:

```bash
# Install Python 3.9 (Run this command alone, and press `y` when indicated)
sudo apt install python3.9

# Set up `python` symlink (Do not miss out the `1` at the end)
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Download `distutil` and `pip`
sudo apt install python3.9-distutils
curl https://bootstrap.pypa.io/get-pip.py | sudo python3.9

# Set up `pip` symlink (Do not miss out the `1` at the end)
sudo update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.9 1

# Download Python3.9 venv
sudo apt install python3.9-venv
```

### Whether on the AWS Instance or locally, set up package environment:

**MacOS/Linux(AWS Instance)**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
venv/Scripts/activate
```

### Whether on the AWS Instance or locally, run this for predictions:

```Bash
# Clone the repo and change directory
git clone https://github.com/AY2425S1-DSA4262-gene-ius/Genomics-Project.git
cd Genomics-Project

# Install packages
pip install -r requirements.txt

# Run the prediction
python -m make_predictions --data_file_path data/sample_data.json.gz --model_path models/Histogram-based_Gradient_Boosting.joblib --standard_scaler_path artifacts/standard_scaler.joblib --pca_path artifacts/pca.joblib --output_file_name sample_data_predictions.csv

# If you`re on linux, read the output in the terminal
head predictions/sample_data_predictions.csv
```

Back to the README...

## System Requirements

Our workflow requires **Python 3.9** or later. Please ensure your Python version is correct before proceeding.

**AWS Instance:**

When initialising your instance, kindly stick to `EBSVolumeSize` of at least `100` and `InstanceType` of at least `t3.medium`.

> [!IMPORTANT]
> The instance unfortunately comes with **Python 3.8.10**. As such, please execute the commands below sequentially to upgrade it to **Python 3.9**:

```bash
# Install Python 3.9
sudo apt install python3.9

# Point `python` symlink to the installed Python 3.9 (Do not miss out the `1` at the end)
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
```

**Local Machine:**

Ensure that you have downloaded **Python 3.9** or later in your machine.

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

**AWS Instance:**

Unfortunately, `pip` does not come with the instance. Execute the commands below sequentially:

```bash
# Download `distutil` and `pip`
sudo apt install python3.9-distutils
curl https://bootstrap.pypa.io/get-pip.py | sudo python3.9

# Set up `pip` symlink to point to Python 3.9's `pip` (Do not miss out the `1` at the end)
sudo update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.9 1
```

**Local Machine:**

If Python is already installed, `pip` should be included automatically.

### Setup Local Environment

We strongly encourage you to utilise a local environment for package installation before proceeding with the steps.

Refer to the following links for possible setup of the local environment:

-   [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
-   [venv](https://docs.python.org/3/library/venv.html)

To quickly set up an environment, you may simply run the following:

**MacOS**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows/Linux(AWS Instance)**

```bash
python -m venv venv
venv/Scripts/activate
```

### Installing packages

While in `Genomics-Project` folder, run:

```bash
pip install -r requirements.txt
```

> [!WARNING]
> If you face this error when installing the packages:
> `ERROR: Could not find a version that satisfies the requirement...`
>
> Please refer to the System Requirements section to upgrade Python 3.8.10 to Python 3.9.

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
| `--seed` | `int` | Seed for reproducibility. | `888` |

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
python -m m6a_modifications.modelling --x_train_data_path processed_data/X_train.csv --y_train_data_path processed_data/y_train.csv

# Evaluation
python -m m6a_modifications.evaluation --model_file_path models/Histogram-based_Gradient_Boosting.joblib --data_path processed_data/X_test.csv --data_identity_path processed_data/X_test_identity.csv --labels_path processed_data/y_test.csv
```
