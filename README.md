# Machine Learning Model Selection Project

## Features

- **Comprehensive Exploratory Data Analysis**
- **Object-Oriented Model Implementation**
- **Multiple Model Comparisons**
- **Hyperparameter Tuning**
- **Model Performance Visualization**
- **Detailed Model Selection Justification**

## Models Implemented

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

## Technical Implementation

### 1. **Exploratory Data Analysis (EDA)**
File: `eda.ipynb`

- Detailed data exploration

### 2. **Model Implementation**
File: `model_.py`

- Object-oriented implementation with base and specific model classes:
  - **Data Loading**: Flexible functionality to load datasets.
  - **Preprocessing**: Comprehensive pipeline for data cleaning, encoding, and scaling.
  - **Training**: Easy-to-use functions for training models.
  - **Evaluation**: In-depth testing and performance evaluation.

### 3. **Model Selection**
File: `model_selection.ipynb`

- Model training and performance comparison
- Performance visualization:
  - **ROC Curves**
  - **Confusion Matrices**
- Hyperparameter tuning for optimal model settings
- Final model selection with detailed justification

## Results

After evaluation, the **Gradient Boosting Classifier** was selected as the final model based on the following metrics:

- **ROC AUC Score**: 0.703
- **Average Precision**: 0.796

## Requirements

To run the project, ensure you have the following installed:

- Python 3.x
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- Jupyter Notebook

