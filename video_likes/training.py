import pickle

import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import KFold, train_test_split

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import Pipeline


def save_model(model, path):

    with open(path, 'wb') as write_buffer:

        pickle.dump(model, write_buffer)


def load_model(path):

    with open(path, 'rb') as read_buffer:

        model = pickle.load(read_buffer)

    return model


def evaluate(model, data):

    x, y = data

    y_hat = model.predict(x)

    mse = mean_squared_error(y, y_hat)
    r2 = r2_score(y, y_hat)

    return mse, r2


def fit_model(model, train_data, val_data, return_r2=True):

    x_train, y_train = train_data
    x_val, y_val = val_data

    model.fit(x_train, y_train)

    mse, r2 = evaluate(model, (x_val, y_val))

    if return_r2:

        return mse, r2

    return mse


def cv_fit(model, x, y, normalized=True):

    if normalized:

        model = Pipeline([('scaler', RobustScaler()), ('model', model)])
    
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)

    loss = []

    for i, (train_idx, val_idx) in enumerate(kfold.split(x, y)):

        train_data = x[train_idx], y[train_idx]
        val_data = x[val_idx], y[val_idx]

        mse, r2 = fit_model(model, train_data, val_data)

        print(f'Fold = {i + 1}, mse = {mse}, r2: {r2}')

        loss.append(mse)

    return np.mean(loss)


def fit_split(model, x, y, normalized=True, save=False, path=None):

    if normalized:

        model = Pipeline([('scaler', RobustScaler()), ('model', model)])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    mse, r2 = fit_model(model, (x_train, y_train), (x_test, y_test))

    if save:

        save_model(model, path)

    return mse, r2
