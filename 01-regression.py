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


if __name__ == '__main__':
    # (1) Regression task prediction
    # dm01_Regression_pred()
    # (2) Regression task save and load model
    dm02_Regression_save_load()
