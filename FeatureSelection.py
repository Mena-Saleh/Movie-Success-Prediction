import pickle
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression  # You can replace this with any regression model of your choice

def rfe_select(X, y, n = 20, estimator = LinearRegression()):
    # Initialize the RFE selector
    rfe = RFE(estimator = estimator, n_features_to_select = n)

    # Fit the RFE selector to the data
    rfe.fit(X, y)

    # Get the boolean mask of selected features
    selected_features = rfe.support_

    # Get the feature ranking (1 for selected features, 0 for eliminated features)
    ranked_features = rfe.ranking_

    # Save the selected features using pickle
    with open('Pickled/selected_features.pkl', 'wb') as file:
        pickle.dump(selected_features, file)

    return selected_features, ranked_features

# Example usage:
# selected_features, ranked_features = select_features_with_rfe(X_train, y_train)
# X_train_selected = X_train.loc[:, selected_features]
# X_test_selected = X_test.loc[:, selected_features]
