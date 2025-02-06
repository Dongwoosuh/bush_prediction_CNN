import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def predict_on_grid(model, poly, grid_points):
    """
    Predict using the polynomial regression model on the grid points.
    Args:
        model: Trained LinearRegression model.
        poly: PolynomialFeatures instance.
        grid_points: Grid points (2D numpy array).
    Returns:
        Predicted values on the grid.
    """
    grid_points_poly = poly.transform(grid_points)  # Expand grid points to polynomial terms
    return model.predict(grid_points_poly)

def polynomial_regression(X, Z, degree, prev_model=None, prev_poly=None):

    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)  # Expand input features to polynomial terms

    # Linear regression with positive constraint
    model = LinearRegression(positive=True, fit_intercept=False)
    # model = LinearRegression(fit_intercept=False)

    try:
        model.fit(X_poly, Z)  # Fit linear regression on expanded features
    except RuntimeError as e:
        # print(f"Fitting failed for degree {degree}: {e}")
        if prev_model is not None and prev_poly is not None:
            # print("Using previous model and polynomial features as fallback.")
            return prev_model, prev_poly
        else:
            raise RuntimeError("No previous model available to fallback.")
    
    return model, poly

def loocv_optimization(X, Z, max_degree=6):

    from sklearn.exceptions import ConvergenceWarning
    import warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    loo = LeaveOneOut()
    errors = []
    valid_degrees = []

    for degree in range(2, max_degree + 1):
        try:
            mse_list = []
            for train_index, test_index in loo.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Z_train, Z_test = Z[train_index], Z[test_index]

                # Train the model
                model, poly = polynomial_regression(X_train, Z_train, degree)
                # Predict on the test set
                Z_pred = predict_on_grid(model, poly, X_test)
                # Compute the mean squared error
                mse_list.append(mean_squared_error(Z_test, Z_pred))

            # Average MSE for this degree
            errors.append(np.mean(mse_list))
            valid_degrees.append(degree)

        except Exception as e:
            print(f"Error occurred for degree {degree}: {str(e)}")
            continue

    if not valid_degrees:
        raise ValueError("All degrees failed during LOOCV.")

    # Find the degree with the lowest error
    optimal_degree = valid_degrees[np.argmin(errors)]
    return optimal_degree

def main(X,Z):

    optimal_degree = loocv_optimization(X, Z)
    model, poly = polynomial_regression(X, Z, optimal_degree)

    return model
