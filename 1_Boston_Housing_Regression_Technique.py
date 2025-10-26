# ==============================================================
# Linear Regression on Custom Housing Dataset with Auto Target
# ==============================================================

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load dataset from CSV
# -------------------------------
file_path = "1housing.csv"  # your CSV file path
df = pd.read_csv(file_path)
print(f"✅ Dataset Loaded | Shape: {df.shape}")
print(df.head())

# -------------------------------
# 2. Detect numeric columns for regression
# -------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    raise ValueError("No numeric columns found for regression.")

# -------------------------------
# 3. Auto-select target (highest variance) or manual selection
# -------------------------------
target_col = df[numeric_cols].var().idxmax()
print(f"✅ Automatically detected target column (highest variance): {target_col}")

# Optional: allow manual override
user_input = input("Do you want to manually set the target column? (y/n): ").strip().lower()
if user_input == 'y':
    while True:
        manual_target = input(f"Enter target column from {numeric_cols}: ").strip()
        if manual_target in numeric_cols:
            target_col = manual_target
            break
        else:
            print("Invalid column name. Try again.")
print(f"✅ Using target column: {target_col}")

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# -------------------------------
# 4. Exploratory Data Analysis (EDA)
# -------------------------------
print(df.describe())

# Correlation with target
corr_matrix = df.corr()
print("\nCorrelation with target column:")
print(corr_matrix[target_col].sort_values(ascending=False))

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title(f'Correlation matrix of Housing Dataset (Target: {target_col})')
plt.show()

# -------------------------------
# 5. Feature Selection (top 5 correlated with target)
# -------------------------------
corr_with_target = corr_matrix[target_col].drop(target_col).abs()
selected_features = corr_with_target.sort_values(ascending=False).head(5).index.tolist()
print(f"✅ Selected features: {selected_features}")

X = df[selected_features]

# -------------------------------
# 6. Split data into training and testing sets
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 7. Train Linear Regression model
# -------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on test set
y_pred = lr.predict(X_test)

# -------------------------------
# 8. Evaluate model performance
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n--- PERFORMANCE METRICS ---")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R-squared (R2) Score: {r2:.3f}")

# Plot Actual vs Predicted values
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred, line_kws={"color": "red"})
plt.xlabel(f'Actual {target_col}')
plt.ylabel(f'Predicted {target_col}')
plt.title('Actual vs Predicted House Prices with Regression Line')
plt.show()

# -------------------------------
# 9. Interactive Prediction (fixed warning)
# -------------------------------
print(f"\nEnter values for the following features to predict the house price ({target_col}):")
input_values = []
for feature in selected_features:
    while True:
        try:
            val = float(input(f"{feature}: "))
            input_values.append(val)
            break
        except ValueError:
            print("Please enter a numeric value.")

# Convert input to DataFrame to keep feature names
input_df = pd.DataFrame([input_values], columns=selected_features)
predicted_price = lr.predict(input_df)[0]

print(f"\nPredicted House Price ({target_col}): ${predicted_price:,.2f}")




# OUTPUT:
# ✅ Dataset Loaded | Shape: (506, 14)
#       crim    zn  indus  chas    nox     rm   age     dis  rad  tax  ptratio       b  lstat  medv
# 0  0.00632  18.0   2.31     0  0.538  6.575  65.2  4.0900    1  296     15.3  396.90   4.98  24.0
# 1  0.02731   0.0   7.07     0  0.469  6.421  78.9  4.9671    2  242     17.8  396.90   9.14  21.6
# 2  0.02729   0.0   7.07     0  0.469  7.185  61.1  4.9671    2  242     17.8  392.83   4.03  34.7
# 3  0.03237   0.0   2.18     0  0.458  6.998  45.8  6.0622    3  222     18.7  394.63   2.94  33.4
# 4  0.06905   0.0   2.18     0  0.458  7.147  54.2  6.0622    3  222     18.7  396.90   5.33  36.2
# ✅ Automatically detected target column (highest variance): tax
# Do you want to manually set the target column? (y/n): n
# ✅ Using target column: tax
#              crim          zn       indus        chas         nox  ...         tax     ptratio           b       lstat        medv
# count  506.000000  506.000000  506.000000  506.000000  506.000000  ...  506.000000  506.000000  506.000000  506.000000  506.000000
# mean     3.613524   11.363636   11.136779    0.069170    0.554695  ...  408.237154   18.455534  356.674032   12.653063   22.532806
# std      8.601545   23.322453    6.860353    0.253994    0.115878  ...  168.537116    2.164946   91.294864    7.141062    9.197104
# min      0.006320    0.000000    0.460000    0.000000    0.385000  ...  187.000000   12.600000    0.320000    1.730000    5.000000
# 25%      0.082045    0.000000    5.190000    0.000000    0.449000  ...  279.000000   17.400000  375.377500    6.950000   17.025000
# 50%      0.256510    0.000000    9.690000    0.000000    0.538000  ...  330.000000   19.050000  391.440000   11.360000   21.200000
# 75%      3.677083   12.500000   18.100000    0.000000    0.624000  ...  666.000000   20.200000  396.225000   16.955000   25.000000
# max     88.976200  100.000000   27.740000    1.000000    0.871000  ...  711.000000   22.000000  396.900000   37.970000   50.000000

# [8 rows x 14 columns]

# Correlation with target column:
# tax        1.000000
# rad        0.910228
# indus      0.720760
# nox        0.668023
# crim       0.582764
# lstat      0.543993
# age        0.506456
# ptratio    0.460853
# chas      -0.035587
# rm        -0.292048
# zn        -0.314563
# b         -0.441808
# medv      -0.468536
# dis       -0.534432
# Name: tax, dtype: float64
# ✅ Selected features: ['rad', 'indus', 'nox', 'crim', 'lstat']

# --- PERFORMANCE METRICS ---
# Mean Squared Error (MSE): 3512.890
# R-squared (R2) Score: 0.887

# Enter values for the following features to predict the house price (tax):
# rad: 85
# indus: 56
# nox: 63
# crim: 66
# lstat: 15

# Predicted House Price (tax): $2,700.10