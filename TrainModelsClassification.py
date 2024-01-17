import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import Preprocessing as pp
import Models as md
import FeatureSelection as fe
from sklearn.preprocessing import LabelEncoder
import pickle

# Read data
df = pd.read_csv('movies-classification-dataset.csv')

# Separate the DataFrame into X and y
X = df.drop('Rate', axis=1)
y = df['Rate']

# Encode y and remember encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Save the label encoder using pickle
with open('label_encoder_y.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

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
selected_features, ranked_features = fe.rfe_select(X_train, y_train, n=50)
X_train_selected = X_train.loc[:, :] # No feature selection yields better results somehow
X_test_selected = X_test.loc[:, :]

# Models training
#1 Logistic Regression:
print("#1 Logistic Regression:")
md.logistic_regression(X_train_selected, y_train, X_test_selected, y_test)

#2 SVM with Gaussian Kernel:
print("#2 SVM:")
md.svm(X_train_selected, y_train, X_test_selected, y_test, C = 1, kernel = 'rbf')

#3 Decision Tree:
print("#3 Decision Tree:")
md.decision_tree(X_train_selected, y_train, X_test_selected, y_test, max_depth= 20)

#4 Random Forest:
print("#4 Random Forest:")
md.random_forest(X_train_selected, y_train, X_test_selected, y_test, max_depth= 50, n_estimators= 300)

#5 Gradient Boosting Classifier:
print("#5 Gradient Boosting Classifier:")
md.gradient_boosting(X_train_selected, y_train, X_test_selected, y_test, max_depth= 5, n_estimators= 100)

#6 Naive Bayes:
print("#6 Naive Bayes:")
md.naive_bayes(X_train_selected, y_train, X_test_selected, y_test)

#7 SGD Classifier:
print("#7 SGD Classifier:")
md.sgd_classifier(X_train_selected, y_train, X_test_selected, y_test)
