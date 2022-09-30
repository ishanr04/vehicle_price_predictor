import pickle

import hyperopt
import pandas as pd
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, rand
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 12)

cars = pd.read_csv("/Users/ishanranasinghe/Desktop/Filtered_Car_Dataset.csv")

make_col_enc = pickle.load(open('Data_Encoding/make_col_enc.pkl', 'rb'))
cars['make'] = make_col_enc.transform(cars['make'])

model_col_enc = pickle.load(open('Data_Encoding/model_col_enc.pkl', 'rb'))
cars['model'] = model_col_enc.transform(cars['model'])

cars_pre = pd.get_dummies(cars, columns=['transmission', 'fuel_type'])

X = cars_pre.drop('price', axis=1)
Y = cars_pre['price']

a_pre = np.array(X)
b_pre = np.array(Y)

a_pre_train, a_pre_test, b_pre_train, b_pre_test = train_test_split(a_pre, b_pre, test_size=0.2)

space = {'n_estimators': hp.choice('n_estimators', [10, 50, 200, 750]),
         'max_depth': hp.uniform("max_depth", 5, 35),
         'gamma': hp.uniform('gamma', 1, 9),
         'reg_alpha': hp.uniform('reg_alpha', 20, 50),
         'reg_lambda': hp.uniform('reg_lambda', 0, 1),
         'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
         'min_child_weight': hp.uniform('min_child_weight', 0, 10)
         }


def objective(space):
    model = XGBRegressor(
        n_estimators=space['n_estimators'], max_depth=int(space['max_depth']), gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']), reg_lambda=int(space['reg_lambda']),
        colsample_bytree=int(space['colsample_bytree']),
        min_child_weight=int(space['min_child_weight']))
    accuracy = cross_val_score(model, a_pre_train, b_pre_train, cv=4).mean()
    print("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}


trials = Trials()

best_hyperparams = fmin(fn=objective,
                        space=space,
                        algo=rand.suggest,
                        max_evals=100,
                        trials=trials)

print("The best hyper parameters are : ", "\n")
print(best_hyperparams)
