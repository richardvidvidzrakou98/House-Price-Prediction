# House Prices Prediction

A machine learning project that predicts residential house prices using the **House Prices: Advanced Regression Techniques** dataset from Kaggle. This project demonstrates the complete machine learning workflow—from data exploration and preprocessing to model training, evaluation, and visualization—using **Linear Regression**.

---

## Overview

The objective of this project is to build a regression model capable of predicting the selling price of residential homes based on their characteristics. The dataset contains **79 explanatory variables** describing various aspects of houses in Ames, Iowa.

**Goal:** Predict the **SalePrice** of each house.

---

## Dataset

Download the dataset from the Kaggle competition:

**House Prices: Advanced Regression Techniques**

Files used:

- `train.csv` – Training dataset
- `test.csv` – Test dataset
- `sample_submission.csv` – Submission format

Target Variable:

- **SalePrice** – The final selling price of a house.

---

## Technologies Used

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Project Structure

```
House-Price-Prediction/
│
├── app/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── house_price_prediction.ipynb
│  
│
├── images/
│   ├── predicted_vs_actual.png
│   ├── residual_plot.png
│   └── feature_importance.png
│
├── house-prices-advanced-regression
└── README.md
```

---

# Project Workflow

## 1. Load & Explore the Data

The dataset is loaded using Pandas and inspected to understand its structure.

Tasks performed:

- Load the training dataset
- Check dataset dimensions
- Display summary statistics
- Identify missing values
- Explore feature distributions

Example output:

```
Dataset Shape:
(1460, 81)

Top Missing Values

PoolQC          1453
Alley           1369
LotFrontage      259
```

---

## 2. Feature Engineering & Preprocessing

### Handle Missing Values

Missing numerical values are replaced with the median, while remaining incomplete rows are removed.

Example:

```python
df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
df.dropna(inplace=True)
```

### Encode Categorical Features

Categorical variables are converted into numerical values using one-hot encoding.

```python
df = pd.get_dummies(df, drop_first=True)
```

---

### Select Features

For this baseline model, six important features are selected:

- OverallQual
- GrLivArea
- GarageCars
- TotalBsmtSF
- FullBath
- YearBuilt

```python
features = [
    'OverallQual',
    'GrLivArea',
    'GarageCars',
    'TotalBsmtSF',
    'FullBath',
    'YearBuilt'
]

X = df[features]
y = df['SalePrice']
```

---

### Train-Test Split

The data is divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

## 3. Train the Linear Regression Model

### Feature Scaling

Features are standardized before training.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### Train Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
```

### How It Works

- **StandardScaler** normalizes feature values so that variables with larger scales do not dominate the model.
- **Linear Regression** fits a linear equation to the data.
- **model.fit()** learns the relationship between the selected features and house prices using **Ordinary Least Squares (OLS)**.
- **model.predict()** estimates house prices for unseen data.

The optimization objective is to minimize the **sum of squared errors** between actual and predicted prices.

---

## 4. Evaluate the Model

Model performance is measured using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### Example Results

| Metric | Value |
|---------|------:|
| RMSE | ~$27,842 |
| MAE | ~$18,213 |
| R² Score | 0.8247 |

### Interpretation

| R² Score | Meaning |
|----------|---------|
| > 0.80 | Strong model |
| 0.60 – 0.79 | Moderate model |
| < 0.60 | Needs improvement |

---

## 5. Visualize Results

### 📈 Predicted vs Actual Prices

This plot compares predicted prices with actual house prices.

**Interpretation**

- Points close to the red diagonal indicate accurate predictions.
- Larger deviations represent prediction errors.

---

### 📉 Residual Plot

Residuals represent the difference between actual and predicted prices.

**Interpretation**

- Random scatter around zero indicates a healthy model.
- Visible patterns may indicate systematic bias.

---

### 📊 Feature Importance

Linear Regression coefficients show how each feature influences the predicted house price.

**Interpretation**

- Positive coefficient → increases predicted price.
- Negative coefficient → decreases predicted price.
- Larger absolute values indicate greater influence.

---

# 📋 Expected Output

After running the notebook, you should see:

```
Dataset Shape:
(1460, 81)

Missing Values

PoolQC       1453
Alley        1369
LotFrontage   259

Training complete!

RMSE : $27,842
MAE  : $18,213
R²   : 0.8247
```

You should also generate:

- Predicted vs Actual plot
- Residual plot
- Feature Importance plot

---

# 📈 Future Improvements

Possible enhancements include:

- Use all 79 available features
- Apply a logarithmic transformation to `SalePrice`
- Train Ridge Regression
- Train Lasso Regression
- Perform 5-fold Cross Validation
- Experiment with Random Forest Regressor
- Build a Streamlit web application for live house price predictions

---

# Getting Started

Clone the repository:

```bash
git clone https://github.com/yourusername/House-Price-Prediction.git
```

Navigate to the project directory:

```bash
cd House-Price-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
house_price_prediction.ipynb
```

Run all cells to reproduce the results.

---

# References

- Kaggle – House Prices: Advanced Regression Techniques
- Scikit-learn Documentation
- Pandas Documentation
- NumPy Documentation
- Matplotlib Documentation

---

