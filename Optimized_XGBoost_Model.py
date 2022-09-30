import pickle
import pandas as pd
import numpy as np
import seaborn as sns # Data Visualization
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
pd.set_option('display.width',300)
pd.set_option('display.max_columns',12)

cars = pd.read_csv("/Users/ishanranasinghe/Desktop/Filtered_Car_Dataset.csv")

make_col_enc = pickle.load(open('Data_Encoding/make_col_enc.pkl','rb'))
cars['make'] = make_col_enc.transform(cars['make'])

model_col_enc = pickle.load(open('Data_Encoding/model_col_enc.pkl','rb'))
cars['model'] = model_col_enc.transform(cars['model'])


cars_pre = pd.get_dummies(cars, columns=['transmission','fuel_type'])
X = cars_pre.drop('price',axis=1)
Y = cars_pre['price']

a_pre = np.array(X)
b_pre = np.array(Y)

print(a_pre)
a_pre_train , a_pre_test, b_pre_train, b_pre_test = train_test_split(a_pre,b_pre, test_size= 0.2)

Opt_XGBoost_model = XGBRegressor(
                            n_estimators=200, max_depth=30, gamma=4.444295681065996,
                            colsample_bytree=0.8530663201574928, reg_alpha=30, reg_lambda=0.5,
                            min_child_weight=2, learning_rate=0.1, random_state=3
                    )


Opt_XGBoost_model.fit(a_pre_train,b_pre_train)

accuracy_training = Opt_XGBoost_model.score(a_pre_train,b_pre_train)
print('Training Accuracy:',accuracy_training*100,'%')

b_train_predict = Opt_XGBoost_model.predict(a_pre_train)
print('MAE:', mean_absolute_error(b_pre_train, b_train_predict))
print('R2 score', r2_score(b_pre_train, b_train_predict))


print("-------------------------")

accuracy = Opt_XGBoost_model.score(a_pre_test,b_pre_test)
print('Testing Accuracy:',accuracy*100,'%')

b_predict = Opt_XGBoost_model.predict(a_pre_test)
print('MAE:', mean_absolute_error(b_pre_test, b_predict))
print('R2 score', r2_score(b_pre_test, b_predict))


fig = sns.regplot(x=b_predict,y=b_pre_test,color='brown', marker="^")
fig.set(title="Optimized XGBoost Model", xlabel = "Predicted price")
plt.show()

pickle.dump(Opt_XGBoost_model, open('Opt_XGBoostModel.pkl','wb'))



