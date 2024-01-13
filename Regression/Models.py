import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Measures R2 score and MSE
def calculate_metrics(y_true, y_pred):
    # Calculate R-squared (R2) score
    r2 = r2_score(y_true, y_pred)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)

    # Print metrics
    print(f"R2 Score: {r2}")
    print(f"MSE: {mse}")
    
    return r2, mse

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

def simple_linear_regression(X_train, y_train, X_test, y_test):
    # Build the simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2, mse = calculate_metrics(y_test, y_pred)

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
    r2, mse = calculate_metrics(y_test, y_pred)

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
    r2, mse = calculate_metrics(y_test, y_pred)

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
    r2, mse = calculate_metrics(y_test, y_pred)

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
    r2, mse = calculate_metrics(y_test, y_pred)

    # Visualize predictions
    visualize_predictions(y_test, y_pred)

    # Save the trained model using pickle
    with open("Pickled/sgd_regression.pkl", "wb") as f:
        pickle.dump(model, f)

    return r2, mse

def build_mlp_regression(X_train, y_train, X_test, y_test, hidden_layers=[64, 32], activation='relu', epochs=100):
    # Define the model
    model = tf.keras.Sequential()

    # Add input layer
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))

    # Add hidden layers
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation=activation))

    # Add output layer (1 unit for regression)
    model.add(tf.keras.layers.Dense(1))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= 0.1),
                  loss='mean_squared_error')  # For regression

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_split=0.2)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2, mse = calculate_metrics(y_test, y_pred)

    # Visualize predictions
    visualize_predictions(y_test, y_pred)

    # Save the trained model using TensorFlow SavedModel format
    model.save("tf_mlp_regression")

    return r2, mse


# Example usage:
# r2_score, mse_score = simple_linear_regression(X_train, y_train, X_test, y_test)
