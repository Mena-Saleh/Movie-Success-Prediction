import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor, SGDClassifier, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# Measures R2 score and MSE for Regression models
def calculate_metrics_regression(y_true, y_pred):
    # Calculate R-squared (R2) score
    r2 = r2_score(y_true, y_pred)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)

    # Print metrics
    print(f"R2 Score: {r2}")
    print(f"MSE: {mse}")
    
    return r2, mse

# Draws confusion matrix as well as important metrics like accuracy, recall etc.
def calculate_metrics_classification(y_true, y_pred):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate other metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Print the metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.show()

# Plot predictions vs actual results.
def visualize_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    plt.grid(True)
    plt.show()

# Models definition


## Regression

def simple_linear_regression(X_train, y_train, X_test, y_test):
    # Build the simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2, mse = calculate_metrics_regression(y_test, y_pred)

    # Visualize predictions
    visualize_predictions(y_test, y_pred)

    # Save the trained model using pickle
    with open("Pickled/simple_linear_regression.pkl", "wb") as f:
        pickle.dump(model, f)
    return r2, mse

def polynomial_regression(X_train, y_train, X_test, y_test, degree = 2):
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Build the polynomial regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_poly)

    # Calculate metrics
    r2, mse = calculate_metrics_regression(y_test, y_pred)

    # Visualize predictions
    visualize_predictions(y_test, y_pred)

    # Save the trained model using pickle
    with open("Pickled/polynomial_regression.pkl", "wb") as f:
        pickle.dump(model, f)

    return r2, mse

def random_forest_regression(X_train, y_train, X_test, y_test,n_estimators=100, max_depth=None):
    # Build the Random Forest regression model
    model = RandomForestRegressor(n_estimators= n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2, mse = calculate_metrics_regression(y_test, y_pred)

    # Visualize predictions
    visualize_predictions(y_test, y_pred)

    # Save the trained model using pickle
    with open("Pickled/random_forest_regression.pkl", "wb") as f:
        pickle.dump(model, f)

    return r2, mse

def gradient_boosting_regression(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, max_depth=3):
    # Build the Gradient Boosting regression model
    model = GradientBoostingRegressor(n_estimators= n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2, mse = calculate_metrics_regression(y_test, y_pred)

    # Visualize predictions
    visualize_predictions(y_test, y_pred)

    # Save the trained model using pickle
    with open("Pickled/gradient_boosting_regression.pkl", "wb") as f:
        pickle.dump(model, f)

    return r2, mse

def sgd_regression(X_train, y_train, X_test, y_test, learning_rate = "adaptive", max_iter= 1000):
    # Build the SGD regression model
    model = SGDRegressor(learning_rate=learning_rate, max_iter= max_iter)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2, mse = calculate_metrics_regression(y_test, y_pred)

    # Visualize predictions
    visualize_predictions(y_test, y_pred)

    # Save the trained model using pickle
    with open("Pickled/sgd_regression.pkl", "wb") as f:
        pickle.dump(model, f)

    return r2, mse


## Classification


def logistic_regression(X_train, y_train, X_test, y_test, max_iter = 1000):
    model = LogisticRegression(max_iter= max_iter)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = calculate_metrics_classification(y_test, y_pred)
    with open("Pickled/logistic_regression.pkl", "wb") as f:
        pickle.dump(model, f)
    return metrics

def svm(X_train, y_train, X_test, y_test, C = 1, kernel='rbf'):
    model = SVC(kernel=kernel, C=C, gamma='scale')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = calculate_metrics_classification(y_test, y_pred)
    with open("Pickled/svm.pkl", "wb") as f:
        pickle.dump(model, f)
    return metrics

def decision_tree(X_train, y_train, X_test, y_test, max_depth=5, min_samples_split=2, min_samples_leaf=1):
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = calculate_metrics_classification(y_test, y_pred)
    with open("Pickled/decision_tree.pkl", "wb") as f:
        pickle.dump(model, f)
    return metrics

def random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=5):
    model = RandomForestClassifier(n_estimators= n_estimators, max_depth= max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = calculate_metrics_classification(y_test, y_pred)
    with open("Pickled/random_forest_classification.pkl", "wb") as f:
        pickle.dump(model, f)
    return metrics

def gradient_boosting(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, max_depth=3):
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate= learning_rate, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = calculate_metrics_classification(y_test, y_pred)
    with open("Pickled/gradient_boosting.pkl", "wb") as f:
        pickle.dump(model, f)
    return metrics

def naive_bayes(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = calculate_metrics_classification(y_test, y_pred)

    with open("Pickled/naive_bayes.pkl", "wb") as f:
        pickle.dump(model, f)
    return metrics

def sgd_classifier(X_train, y_train, X_test, y_test, max_iter = 1000):
    model = SGDClassifier(max_iter=max_iter, tol=1e-3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = calculate_metrics_classification(y_test, y_pred)
    with open("Pickled/sgd_classifier.pkl", "wb") as f:
        pickle.dump(model, f)
    return metrics



# Example usage:
# r2_score, mse_score = simple_linear_regression(X_train, y_train, X_test, y_test)
