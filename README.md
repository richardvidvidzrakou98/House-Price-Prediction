# House Prices Prediction

A machine learning project that predicts residential house prices using the **House Prices: Advanced Regression Techniques** dataset from Kaggle. The dataset contains 79 explanatory features describing different aspects of homes in Ames, Iowa, with the goal of predicting the final sale price. :contentReference[oaicite:0]{index=0}

## Project Overview

This project demonstrates a complete machine learning workflow for a regression problem, including:

- Data exploration and visualization
- Data cleaning and preprocessing
- Handling missing values
- Feature engineering
- Model training and evaluation
- House price prediction

## Dataset

The dataset includes:

- **train.csv** – Training data with house features and sale prices
- **test.csv** – Test data used for prediction
- **sample_submission.csv** – Sample submission format

Target Variable:

- **SalePrice** – The final selling price of each house

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Workflow

1. Load and explore the dataset
2. Clean and preprocess the data
3. Handle missing values
4. Encode categorical features
5. Train regression models
6. Evaluate model performance
7. Generate predictions for the test dataset

## How to Run

1. Clone the repository.
2. Download the competition dataset from Kaggle.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Run the notebook or Python script:

```bash
jupyter notebook
```

or

```bash
python main.py
```

## Evaluation

Model performance is evaluated using **Root Mean Squared Error (RMSE)**, the official metric for the Kaggle competition. :contentReference[oaicite:1]{index=1}

## Project Structure

```
house-prices/
│── data/
│── notebooks/
│── models/
│── outputs/
│── requirements.txt
│── README.md
```

## References

- Kaggle House Prices: Advanced Regression Techniques Competition
- Ames Housing Dataset
