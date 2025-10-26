import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('2titanic.csv')

# Fix missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop columns with too many missing or irrelevant for now
df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# Map Sex
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode Embarked
embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = pd.concat([df, embarked_dummies], axis=1)
df.drop('Embarked', axis=1, inplace=True)

# Select features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X = df[features]
y = df['Survived']

# Scale numeric features: Age and Fare
scaler = StandardScaler()
X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression with more iterations to ensure convergence
model = LogisticRegression(max_iter=500, solver='lbfgs')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Logistic Regression model: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)
print("\nFeature Coefficients:")
print(coef_df)

# --- User Input for prediction ---
print("\nEnter passenger details to predict survival.")
print("Please enter valid values for each feature.")

input_data = {}

def get_binary_input(feature_name):
    while True:
        val = input(f"{feature_name} (0 or 1): ").strip()
        if val in ['0', '1']:
            return int(val)
        else:
            print("Invalid input. Enter 0 or 1.")

def get_integer_input(feature_name):
    while True:
        val = input(f"{feature_name} (integer): ").strip()
        try:
            intval = int(val)
            return intval
        except ValueError:
            print("Invalid input. Please enter an integer.")

def get_float_input(feature_name):
    while True:
        val = input(f"{feature_name} (number): ").strip()
        try:
            floatval = float(val)
            return floatval
        except ValueError:
            print("Invalid input. Please enter a number.")

for feature in features:
    if feature in ['Sex', 'Embarked_C', 'Embarked_Q', 'Embarked_S']:
        input_data[feature] = get_binary_input(feature)
    elif feature in ['Pclass', 'SibSp', 'Parch']:
        input_data[feature] = get_integer_input(feature)
    else: # Age and Fare
        input_data[feature] = get_float_input(feature)

# Convert input to DataFrame and scale Age and Fare
input_df = pd.DataFrame([input_data])
input_df[['Age', 'Fare']] = scaler.transform(input_df[['Age', 'Fare']])

# Predict survival
survival_pred = model.predict(input_df)[0]
survival_prob = model.predict_proba(input_df)[0][survival_pred]
print("\nPrediction:")
if survival_pred == 1:
    print(f"Survived with probability {survival_prob:.2%}")
else:
    print(f"Did NOT survive with probability {survival_prob:.2%}")




# step

# Short Steps to Run the Code with the Titanic Dataset:

# Install required packages:
# pip install pandas numpy scikit-learn

# Save the code:
# Copy the Python code into a file (e.g., titanic_logistic_regression.py).

# Place the dataset:
# Put titanic.csv in the same folder as your code file.

# Run the script:
# Use the command: python titanic_logistic_regression.py

# View the output:
# Model accuracy and classification report
# Confusion matrix and feature importance
# Prediction prompt—enter passenger details for survival prediction

# except OUTPUT
# PS C:\Documents\Course Project\ML> python .\2_Titanic_Survival_regression_technique.py
# C:\Documents\Course Project\ML\2_Titanic_Survival_regression_technique.py:33: SettingWithCopyWarning: 
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead

# See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
#   X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])
# Accuracy of Logistic Regression model: 0.8101

# Classification Report:
#               precision    recall  f1-score   support

#            0       0.83      0.86      0.84       105
#            1       0.79      0.74      0.76        74

#     accuracy                           0.81       179
#    macro avg       0.81      0.80      0.80       179
# weighted avg       0.81      0.81      0.81       179


# Confusion Matrix:
# [[90 15]
#  [19 55]]

# Feature Coefficients:
#       Feature  Coefficient
# 1         Sex     2.592513
# 6  Embarked_C     0.191362
# 5        Fare     0.124545
# 7  Embarked_Q     0.039624
# 4       Parch    -0.108164
# 8  Embarked_S    -0.230764
# 3       SibSp    -0.293302
# 2         Age    -0.392275
# 0      Pclass    -0.934899

# Enter passenger details to predict survival.
# Please enter valid values for each feature.
# Pclass (integer): 3
# Sex (0 or 1): 0
# Age (number): 45
# SibSp (integer): 5
# Parch (integer): 2
# Fare (number): 5
# Embarked_C (0 or 1): 1
# Embarked_Q (0 or 1): 1
# Embarked_S (0 or 1): 0

# Prediction:
# Did NOT survive with probability 97.91%