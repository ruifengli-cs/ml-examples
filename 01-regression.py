"""
## (1) Regression Task Prediction
# 1. Import dependencies
# 2. Prepare data
# 3. Instantiate linear regression model
# 4. Model training
# 5. Model prediction

x = [[80, 86], [82, 80], [85, 78], [90, 90], [86, 82], [82, 90], [78, 80], [92, 94]]
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]
"""
# 1. Import dependencies
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import joblib


def dm01_Regression_pred():
    # 2. Prepare data: regular exam scores, final exam scores, final grades
    x = [[80, 86], [82, 80], [85, 78], [90, 90], [86, 82], [82, 90], [78, 80], [92, 94]]
    y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]
    # 3. Instantiate linear regression model
    estimator = LinearRegression()
    print('estimator-->', estimator)

    # 4. Model training
    # Print linear regression model parameters coef_ intercept_
    estimator.fit(x, y)
    print('estimator.coef_-->', estimator.coef_)
    print('estimator.intercept_-->', estimator.intercept_)

    # 5. Model prediction
    mypred = estimator.predict([[90, 80]])
    print('mypred-->', mypred)


"""
## (2) Regression Task Model Save and Load
# 1. Import dependencies
# 2. Prepare data
# 3. Instantiate linear regression model
# 4. Model training
# 5. Model prediction
# 6. Save model joblib.dump(estimator, xxpath)
# 7. Load model joblib.load(xxpath)
# 8. Model prediction

x = [[80, 86], [82, 80], [85, 78], [90, 90], [86, 82], [82, 90], [78, 80], [92, 94]]
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

"""
# 1. Import dependencies
import joblib


def dm02_Regression_save_load():
    # 2. Prepare data: regular exam scores, final exam scores, final grades
    x = [[80, 86], [82, 80], [85, 78], [90, 90], [86, 82], [82, 90], [78, 80], [92, 94]]
    y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

    # 3. Instantiate linear regression model
    estimator = LinearRegression()
    print('estimator-->', estimator)

    # 4. Model training
    estimator.fit(x, y)

    # 5. Model prediction
    mypred = estimator.predict([[90, 80]])
    print('mypred-->', mypred)

    # 6. Save model
    print('\nModel save and reload')
    joblib.dump(estimator, './model/mylrmodel01.bin')

    # 7. Load model
    myestimator2 = joblib.load('./model/mylrmodel01.bin')
    print('myestimator2-->', myestimator2)

    # 8. Model prediction
    mypred2 = myestimator2.predict([[90, 80]])
    print('mypred2-->', mypred2)


def dm03_Regression_overfitting():
    """
    Demonstrates how noise in data can lead to overfitting
    Compares simple linear regression vs polynomial regression
    """
    # Generate clean data
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y_clean = 2 * X + 1

    # Add noise to the data
    noise = np.random.normal(0, 2, 100).reshape(-1, 1)
    y_noisy = y_clean + noise

    # Create polynomial features for overfitting demonstration
    poly = PolynomialFeatures(degree=15)
    X_poly = poly.fit_transform(X)

    # Fit models
    # 1. Simple linear regression (underfitting)
    lr_simple = LinearRegression()
    lr_simple.fit(X, y_noisy)

    # 2. Polynomial regression (overfitting)
    lr_poly = LinearRegression()
    lr_poly.fit(X_poly, y_noisy)

    # Generate points for plotting
    X_plot = np.linspace(0, 10, 1000).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)

    # Make predictions
    y_pred_simple = lr_simple.predict(X_plot)
    y_pred_poly = lr_poly.predict(X_plot_poly)

    # Calculate MSE
    mse_simple = mean_squared_error(y_noisy, lr_simple.predict(X))
    mse_poly = mean_squared_error(y_noisy, lr_poly.predict(X_poly))

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot 1: Simple Linear Regression
    plt.subplot(1, 2, 1)
    plt.scatter(X, y_noisy, color='blue', alpha=0.5, label='Noisy Data')
    plt.plot(X_plot, y_pred_simple, color='red', label='Simple Linear Regression')
    plt.plot(X_plot, 2 * X_plot + 1, color='green', linestyle='--', label='True Pattern')
    plt.title(f'Simple Linear Regression\nMSE: {mse_simple:.2f}')
    plt.legend()

    # Plot 2: Polynomial Regression (Overfitting)
    plt.subplot(1, 2, 2)
    plt.scatter(X, y_noisy, color='blue', alpha=0.5, label='Noisy Data')
    plt.plot(X_plot, y_pred_poly, color='red', label='Polynomial Regression')
    plt.plot(X_plot, 2 * X_plot + 1, color='green', linestyle='--', label='True Pattern')
    plt.title(f'Polynomial Regression (Overfitting)\nMSE: {mse_poly:.2f}')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print model coefficients
    print("\nSimple Linear Regression Coefficients:")
    print(f"Slope: {lr_simple.coef_[0][0]:.2f}")
    print(f"Intercept: {lr_simple.intercept_[0]:.2f}")

    print("\nPolynomial Regression Coefficients:")
    print(f"Number of coefficients: {len(lr_poly.coef_[0])}")
    print("First few coefficients:", lr_poly.coef_[0][:5])


if __name__ == '__main__':
    # (1) Regression task prediction
    # dm01_Regression_pred()
    # (2) Regression task save and load model
    # dm02_Regression_save_load()
    # (3) Regression overfitting demonstration
    dm03_Regression_overfitting()
