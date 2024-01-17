import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import Preprocessing as pp
import Models as md
import FeatureSelection as fe

# Read data
df = pd.read_csv('movies-regression-dataset.csv')

# Separate the DataFrame into X and y
X = df.drop('vote_average', axis=1)
y = df['vote_average']

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
df = pp.encode_columns(df, is_train=True) # To fit the encoders on all possible labels prior to splitting (saved in pickle)
X_train = pp.preprocess(X_train)
X_test = pp.preprocess(X_test, is_train= False)


# Save to csv
X_train.to_csv('preprocessed_train.csv', index=False, encoding='utf-8-sig')
X_test.to_csv('preprocessed_test.csv', index=False, encoding='utf-8-sig')

# Feature selection
selected_features, ranked_features = fe.rfe_select(X_train, y_train, n=20)
X_train_selected = X_train.loc[:, :] # No feature selection yields better results somehow, specially in random forests regressors.
X_test_selected = X_test.loc[:, :]

# Models training
print("#1 Simple Linear Regression:")
md.simple_linear_regression(X_train_selected, y_train, X_test_selected, y_test)

print("#2 Polynomial Regression:")
md.polynomial_regression(X_train_selected, y_train, X_test_selected, y_test)

print("#3 Random Forest Regression:")
md.random_forest_regression(X_train_selected, y_train, X_test_selected, y_test)

print("#4 Gradient Boosting Regression:")
md.gradient_boosting_regression(X_train_selected, y_train, X_test_selected, y_test)

print("#5 Stochastic Gradient Descent (SGD) Regression:")
md.sgd_regression(X_train_selected, y_train, X_test_selected, y_test)

