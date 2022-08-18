from multiprocessing import cpu_count

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

import utils
import data
import training


@utils.u_time
def fit_lr():

    x, y = data.load_data(remake=False, as_numpy=True)

    print(x.shape, y.shape)

    model = LinearRegression()

    mse, r2 = training.fit_split(model, x, y, normalized=True, save=True, path='lr.pkl')

    print('=' * 100)
    print('mse = {}, r2: {}'.format(mse, r2))
    print('=' * 100)


@utils.u_time
def fit_knn():

    x, y = data.load_data(remake=False, as_numpy=True)

    print(x.shape, y.shape)

    model = KNeighborsRegressor(n_neighbors=2)

    mse, r2 = training.fit_split(model, x, y, normalized=True, save=True, path='knn.pkl')

    print('=' * 100)
    print('mse = {}, r2: {}'.format(mse, r2))
    print('=' * 100)


@utils.u_time
def fit_rf():

    x, y = data.load_data(remake=False, as_numpy=True)
    y = y.squeeze()

    print(x.shape, y.shape)

    model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=cpu_count())

    mse, r2 = training.fit_split(model, x, y, normalized=True, save=True, path='rf.pkl')

    print('=' * 100)
    print('mse = {}, r2: {}'.format(mse, r2))
    print('=' * 100)


@utils.u_time
def evaluate_model(dataset_path, model_path, remake=True, is_knn=False):

    x, y = data.load_data(dataset_path=dataset_path, remake=remake, as_numpy=True)
    y = y.squeeze()

    print(x.shape, y.shape)

    model = training.load_model(model_path)

    if is_knn:

        x_train, y_train = data.load_data(dataset_path=None, remake=False, as_numpy=True)
        y_train = y_train.squeeze()

        model.fit(x_train, y_train)

    score = training.evaluate(model, (x, y))

    print('=' * 100)
    print('Accuracy: {}'.format(score))
    print('=' * 100)


"""
LR: 
====================================================================================================
CV MSE: 6.477426176 B
Execution Time : 3.53 seconds
====================================================================================================

KNN:
====================================================================================================
CV MSE: 462.279072 M
Execution Time : 25.75 seconds
====================================================================================================

RF:
====================================================================================================
CV MSE: 183.94998474042347 M
Execution Time : 479.15 seconds
====================================================================================================

"""

fit_lr()
# fit_knn()
# fit_rf()

evaluate_model(dataset_path=None, model_path='lr.pkl', remake=False, is_knn=False)
