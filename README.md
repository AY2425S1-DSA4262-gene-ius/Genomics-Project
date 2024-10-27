# Genomics-Project

python -m m6a_modifications.raw_data_preparer --data_file_path data/dataset0.json.gz

python -m m6a_modifications.data_processing --reads_data_path data/dataset0.json.gz.csv --labels_data_path data/data.info.labelled

python -m m6a_modifications.modelling --x_train_data_path data/X_train.csv --y_train_data_path data/y_train.csv

python -m m6a_modifications.evaluation --model_file_path models/Histogram-based_Gradient_Boosting.joblib --x_test_data_path data/X_test.csv --y_test_data_path data/y_test.csv

python -m model_training --data_file_path data/dataset0.json.gz --labels_data_path data/data.info.labelled --output_file_name training_output.csv

python -m make_predictions --data_file_path data/dataset2.json.gz --model_path models/Histogram-based_Gradient_Boosting.joblib --standard_scaler_path artifacts/standard_scaler.joblib --pca_path artifacts/pca.joblib --output_file_name dataset2.csv
