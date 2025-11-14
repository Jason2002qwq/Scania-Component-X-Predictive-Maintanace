# Scania Component X: Predictive Maintenance

This repository provides a complete predictive maintenance pipeline for Scania Component X, demonstrating an end-to-end workflow from data loading and preprocessing to model training, evaluation, and interactive visualizations.\
This is a cource project of *MA 6503 Machine Learning and Data Science, [Nanyang Technological University](https://ntu.edu.sg)*. The original dataset can be found in [Swedish National Data Service](https://researchdata.se/en/catalogue/dataset/2024-34/3).

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Used](#models-used)
- [Custom Cost Function](#custom-cost-function)
- [Results](#results)
- [Visualizations](#visualizations)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

## Project Overview
Predictive maintenance aims to forecast equipment failures before they occur, enabling timely intervention and reducing costly downtime. This project focuses on a specific Scania vehicle component (Component X), leveraging operational sensor readings and vehicle specifications to predict maintenance needs. The core objective is to identify vehicles at risk of failure and categorize the urgency of required maintenance, minimizing operational costs and maximizing vehicle uptime.

## Dataset
The project utilizes a dataset comprising:
- **Operational Readouts**: Time-series sensor data (`train_operational_readouts.csv`, `validation_operational_readouts.csv`, `test_operational_readouts.csv`).
- **Time-to-Event (TTE) / Labels**: Information on vehicle repair times or failure labels (`train_tte.csv`, `validation_labels.csv`, `test_labels.csv`).
- **Vehicle Specifications**: Static characteristics of each vehicle (`train_specifications.csv`, `validation_specifications.csv`, `test_specifications.csv`).

The data is split into training, validation, and test sets.

## Methodology
The pipeline follows a structured approach:
1.  **Data Loading & Merging**: Raw operational readouts, TTE/labels, and vehicle specifications are loaded and merged based on `vehicle_id` and `time_step`.
2.  **Column Identification**: Categorization of columns into `id_cols`, `spec_cols`, `sensor_cols`, and `label_col`.
3.  **Missing Value Handling**: Missing values in sensor columns are filled using forward-fill (`ffill`), backward-fill (`bfill`), and finally zero-fill for any remaining NaNs, grouped by `vehicle_id`.
4.  **Feature Engineering - Aggregated Features**: Time-series sensor data is aggregated per `vehicle_id` to create static features, including mean, standard deviation, minimum, maximum, and last observed values. Time-based features (`time_step_min`, `time_step_max`, `num_observations`) and vehicle specification features are also incorporated.
5.  **Categorical Feature Encoding**: Specification columns (categorical features) are encoded using `LabelEncoder`. Special care is taken to handle unseen categories in validation and test sets by fitting the encoder on all unique categories across all datasets.
6.  **Feature Scaling**: All numerical features are scaled using `StandardScaler` to normalize their ranges, which is crucial for many machine learning algorithms.

## Models Used
Two supervised machine learning models are trained and evaluated:
1.  **Random Forest Classifier**: An ensemble learning method that builds multiple decision trees and merges their predictions to improve accuracy and control overfitting.
2.  **XGBoost Classifier**: An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework.

## Custom Cost Function
A custom cost function is implemented to evaluate model performance, reflecting the real-world costs of maintenance decisions. This function assigns significantly higher penalties to False Negatives (predicting no failure when one occurs) compared to False Positives (predicting failure when none occurs), aligning with the critical nature of predictive maintenance in minimizing expensive unexpected breakdowns.

- False Negatives (missing a real failure) cost: \$200 - \$500
- False Positives (unnecessary maintenance) cost: \$7 - \$10

## Results
The model selection is based on the custom cost function calculated on the validation set. In this execution, the **Random Forest** model emerged as the best performer.

- **Best Model**: Random Forest
- **Validation Cost (Random Forest)**: 57400
- **Validation Cost (XGBoost)**: 58232
- **Test Cost (Best Model)**: 55940

Detailed classification reports and confusion matrices for both models on validation and test sets are available in the notebook.

### Top Features
Feature importance analysis revealed that `time_step_max`, `158_9_last`, `158_9_max`, and other aggregated sensor readings are among the most influential features for predicting component failure.

## Visualizations
The notebook includes a comprehensive suite of visualizations to understand the data, model performance, and maintenance implications:
1.  `figure1_data_overview.png`: Dataset statistics and distributions.
2.  `figure2_correlation_heatmap.png`: Feature correlations.
3.  `figure3_feature_importance.png`: Top 20 most important features.
4.  `figure4_confusion_matrices.png`: Confusion matrices for validation and test sets.
5.  `figure5_roc_curves.png`: ROC curves for binary classification (predicting failure).
6.  `figure6_precision_recall_curves.png`: Precision-Recall curves for binary classification.
7.  `figure7_confidence_distribution.png`: Prediction confidence and urgency score distributions.
8.  `figure8_model_comparison.png`: Comparison of Random Forest and XGBoost based on standard metrics.
9.  `figure9_cost_analysis.png`: Detailed cost analysis including breakdown of False Positive and False Negative costs.
10. `figure10_maintenance_dashboard.png`: A comprehensive dashboard summarizing maintenance priority, risk levels, and sample vehicle time series.

## How to Run
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Jason2002qwq/Scania-Component-X-Predictive-Maintanace/
    cd Scania-Component-X-Predictive-Maintanace
    ```
2.  **Install Dependencies**:
    Ensure you have all required libraries installed. You can typically do this using `pip`:
    ```bash
    pip install -r requirements.txt # (assuming a requirements.txt file exists)
    # Or manually install: pandas numpy scikit-learn xgboost matplotlib seaborn
    ```
3.  **Prepare Data**: Place the `train_operational_readouts.csv`, `train_tte.csv`, `train_specifications.csv`, `validation_operational_readouts.csv`, `validation_labels.csv`, `validation_specifications.csv`, `test_operational_readouts.csv`, `test_labels.csv`, and `test_specifications.csv` files in the specified `datapath` (e.g., `/content/drive/MyDrive/MA6514/data` as used in the notebook).
4.  **Run the Notebook**: Open and run the `Scania_Component_X_Predictive_Maintenance.ipynb` notebook in a compatible environment (e.g., Google Colab, Jupyter).

## Dependencies
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`
- `warnings`
