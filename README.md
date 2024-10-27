# Genomics-Project

python -m modelling.raw_data_preparer --data_file_path data/dataset0.json.gz

python -m modelling.data_processing --reads_data_path data/dataset0.json.gz.csv --labels_data_path data/data.info.labelled

python -m modelling.modelling --x_train_data_path data/X_train.csv --y_train_data_path data/y_train.csv

python -m modelling.evaluation --model_file_path models/Histogram-based_Gradient_Boosting.joblib --x_test_data_path data/X_test.csv --y_test_data_path data/y_test.csv

python -m full_pipeline --data_file_path data/dataset0.json.gz --labels_data_path data/data.info.labelled
